import os
import csv
import numpy as np


GDB_BASE = r"C:\ruta\IVT_Novita.gdb"
GDB_SIN_DEFOR = r"C:\ruta\IVT_sin_defor.gdb"
CARPETA_SALIDA = r"C:\ruta\validacion"
BLOQUE_CELDAS = 50
N_BOOT = 1000
SEMILLA = 2026
NOMBRES_CLASE = {1: "Baja", 2: "Media", 3: "Alta"}
NOMBRES_ESTATUS = {1: "Con título minero", 2: "Solo solicitud minera", 3: "Sin título ni solicitud"}


def _auc_desde_hist(pos, neg):
    p = pos.sum()
    n = neg.sum()
    if p == 0 or n == 0:
        return float("nan")
    neg_debajo = np.cumsum(neg) - neg
    return float((pos * (neg_debajo + 0.5 * neg)).sum() / (p * n))


def _tasa(n, k):
    return float(k) / float(n) * 100.0 if n > 0 else float("nan")


def analizar(iv, cls, dt, ds, loss, area_celda_m2, bloque=BLOQUE_CELDAS, n_boot=N_BOOT, semilla=SEMILLA):
    valido = ~(np.isnan(iv) | np.isnan(cls) | np.isnan(dt) | np.isnan(ds) | np.isnan(loss))
    filas, cols = np.indices(iv.shape)
    bid_full = (filas // bloque) * (cols // bloque + 1000) + (cols // bloque)
    v_iv = np.round(iv[valido], 4)
    v_cls = cls[valido].astype(int)
    v_loss = loss[valido].astype(int)
    v_dt = dt[valido]
    v_ds = ds[valido]
    est = np.where(v_dt == 0, 1, np.where(v_ds == 0, 2, 3))
    bid_raw = bid_full[valido]
    _, bid = np.unique(bid_raw, return_inverse=True)
    nb = int(bid.max()) + 1
    niveles, nivel_idx = np.unique(v_iv, return_inverse=True)
    L = len(niveles)

    def conteo(cat, k):
        m = cat == k
        n = np.bincount(bid[m], minlength=nb)
        p = np.bincount(bid[m], weights=v_loss[m], minlength=nb)
        return np.stack([n, p], axis=1).astype(float)

    cb_cls = {k: conteo(v_cls, k) for k in (1, 2, 3)}
    cb_est = {k: conteo(est, k) for k in (1, 2, 3)}
    idx_h = (bid * L + nivel_idx) * 2 + v_loss
    hist = np.bincount(idx_h, minlength=nb * L * 2).reshape(nb, L, 2).astype(np.int32)

    def estadisticos(sel):
        out = {}
        for k in (1, 2, 3):
            out["cls%d" % k] = cb_cls[k][sel].sum(axis=0)
            out["est%d" % k] = cb_est[k][sel].sum(axis=0)
        h = hist[sel].sum(axis=0)
        out["auc"] = _auc_desde_hist(h[:, 1].astype(float), h[:, 0].astype(float))
        return out

    def derivados(e):
        d = {}
        tot_n = sum(e["cls%d" % k][0] for k in (1, 2, 3))
        tot_p = sum(e["cls%d" % k][1] for k in (1, 2, 3))
        for k in (1, 2, 3):
            n, p = e["cls%d" % k]
            d["inc_cls%d" % k] = _tasa(n, p)
            d["rf_cls%d" % k] = (p / tot_p) / (n / tot_n) if (tot_p > 0 and n > 0) else float("nan")
        for k in (1, 2, 3):
            n, p = e["est%d" % k]
            d["inc_est%d" % k] = _tasa(n, p)
        d["dif_titulo_libre"] = d["inc_est1"] - d["inc_est3"]
        d["rz_titulo_libre"] = d["inc_est1"] / d["inc_est3"] if d["inc_est3"] > 0 else float("nan")
        d["dif_solic_libre"] = d["inc_est2"] - d["inc_est3"]
        d["rz_solic_libre"] = d["inc_est2"] / d["inc_est3"] if d["inc_est3"] > 0 else float("nan")
        d["dif_alta_baja"] = d["inc_cls3"] - d["inc_cls1"]
        d["auc"] = e["auc"]
        d["inc_total"] = _tasa(tot_n, tot_p)
        return d

    base = derivados(estadisticos(np.arange(nb)))
    rng = np.random.RandomState(semilla)
    boots = []
    for _ in range(n_boot):
        boots.append(derivados(estadisticos(rng.randint(0, nb, nb))))
    ic = {}
    for k in base:
        v = np.array([b[k] for b in boots], dtype=float)
        v = v[~np.isnan(v)]
        ic[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"), float("nan"))
    resumen = {
        "n_validas": int(valido.sum()),
        "n_bloques": nb,
        "area_km2": float(valido.sum() * area_celda_m2 / 1e6),
        "n_clase": {k: int(cb_cls[k][:, 0].sum()) for k in (1, 2, 3)},
        "n_estatus": {k: int(cb_est[k][:, 0].sum()) for k in (1, 2, 3)},
    }
    return {"base": base, "ic": ic, "resumen": resumen}


def _f(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%." + str(d) + "f") % x


def escribir_tablas(res, area_celda_m2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    b, ic, rs = res["base"], res["ic"], res["resumen"]
    ar = lambda n: n * area_celda_m2 / 1e6

    def w(nombre, enc, filas):
        with open(os.path.join(out_dir, nombre), "w", newline="", encoding="utf-8-sig") as fh:
            c = csv.writer(fh, delimiter=";")
            c.writerow(enc)
            for f in filas:
                c.writerow(f)

    f1 = []
    for k in (1, 2, 3):
        f1.append([NOMBRES_CLASE[k], rs["n_clase"][k], _f(ar(rs["n_clase"][k])), _f(b["inc_cls%d" % k]),
                   "%s a %s" % (_f(ic["inc_cls%d" % k][0]), _f(ic["inc_cls%d" % k][1])),
                   _f(b["rf_cls%d" % k], 3), "%s a %s" % (_f(ic["rf_cls%d" % k][0], 3), _f(ic["rf_cls%d" % k][1], 3))])
    w("validacion_1_perdida_por_clase_IVT_sin_defor.csv",
      ["clase del IVT sin deforestación", "celdas", "área km2", "% del área con pérdida de bosque 2001-2025",
       "IC 95 % (bootstrap por bloques)", "razón de frecuencias (% de la pérdida / % del área)", "IC 95 %"], f1)

    f2 = []
    for k in (1, 2, 3):
        f2.append([NOMBRES_ESTATUS[k], rs["n_estatus"][k], _f(ar(rs["n_estatus"][k])), _f(b["inc_est%d" % k]),
                   "%s a %s" % (_f(ic["inc_est%d" % k][0]), _f(ic["inc_est%d" % k][1]))])
    f2.append(["Diferencia: con título - libre (puntos porcentuales)", "", "", _f(b["dif_titulo_libre"]),
               "%s a %s" % (_f(ic["dif_titulo_libre"][0]), _f(ic["dif_titulo_libre"][1]))])
    f2.append(["Razón: con título / libre", "", "", _f(b["rz_titulo_libre"], 3),
               "%s a %s" % (_f(ic["rz_titulo_libre"][0], 3), _f(ic["rz_titulo_libre"][1], 3))])
    f2.append(["Diferencia: solo solicitud - libre (puntos porcentuales)", "", "", _f(b["dif_solic_libre"]),
               "%s a %s" % (_f(ic["dif_solic_libre"][0]), _f(ic["dif_solic_libre"][1]))])
    f2.append(["Razón: solo solicitud / libre", "", "", _f(b["rz_solic_libre"], 3),
               "%s a %s" % (_f(ic["rz_solic_libre"][0], 3), _f(ic["rz_solic_libre"][1], 3))])
    w("validacion_2_perdida_por_condicion_minera.csv",
      ["condición", "celdas", "área km2", "% del área con pérdida de bosque 2001-2025", "IC 95 % (bootstrap por bloques)"], f2)

    w("validacion_3_AUC_y_resumen.csv", ["medida", "valor", "IC 95 %"],
      [["AUC del IVT sin deforestación para ordenar celdas con pérdida", _f(b["auc"], 3),
        "%s a %s" % (_f(ic["auc"][0], 3), _f(ic["auc"][1], 3))],
       ["% del área con pérdida (todo el municipio analizado)", _f(b["inc_total"]), ""],
       ["Diferencia Alta - Baja (puntos porcentuales)", _f(b["dif_alta_baja"]),
        "%s a %s" % (_f(ic["dif_alta_baja"][0]), _f(ic["dif_alta_baja"][1]))],
       ["celdas válidas", rs["n_validas"], ""], ["área analizada km2", _f(rs["area_km2"]), ""],
       ["bloques espaciales", rs["n_bloques"], ""], ["remuestreos bootstrap", N_BOOT, ""]])


def _leer_nan(arcpy, raster):
    import arcpy.sa as sa
    r = arcpy.Raster(raster) if isinstance(raster, str) else raster
    a = arcpy.RasterToNumPyArray(r).astype("float64")
    nulo = arcpy.RasterToNumPyArray(sa.IsNull(r)).astype(bool)
    a[nulo] = np.nan
    return a


def main():
    import arcpy
    import arcpy.sa as sa
    arcpy.CheckOutExtension("Spatial")
    env_prev = (arcpy.env.snapRaster, arcpy.env.extent, arcpy.env.cellSize, arcpy.env.mask)
    try:
        os.makedirs(CARPETA_SALIDA, exist_ok=True)
        p_iv = os.path.join(GDB_SIN_DEFOR, "IVT")
        p_cls = os.path.join(GDB_SIN_DEFOR, "IVT_jenks")
        ref = arcpy.Raster(p_cls)
        area_celda = ref.meanCellWidth * ref.meanCellHeight
        print("Leyendo IVT sin deforestación y su clasificación...")
        iv = _leer_nan(arcpy, p_iv)
        cls = _leer_nan(arcpy, p_cls)
        print("Leyendo distancias, solicitudes y pérdida de bosque (corrida base) y alineándolas a la malla del IVT...")
        arcpy.env.snapRaster = ref
        arcpy.env.extent = ref.extent
        arcpy.env.cellSize = ref.meanCellWidth
        arcpy.env.mask = None
        dt = _leer_nan(arcpy, sa.Raster(os.path.join(GDB_BASE, "dist_titulos")) * 1)
        ds = _leer_nan(arcpy, sa.Raster(os.path.join(GDB_BASE, "dist_solicitudes")) * 1)
        loss = _leer_nan(arcpy, sa.Raster(os.path.join(GDB_BASE, "perdida_prj")) * 1)
        for nombre, a in (("IVT", iv), ("clases", cls), ("dist_titulos", dt), ("dist_solicitudes", ds), ("pérdida", loss)):
            if a.shape != iv.shape:
                raise RuntimeError("La matriz '%s' tiene forma %s y el IVT %s: las mallas no coinciden." % (nombre, a.shape, iv.shape))
        print("Analizando (bootstrap por bloques, puede tardar unos minutos)...")
        res = analizar(iv, cls, dt, ds, loss, area_celda)
        escribir_tablas(res, area_celda, CARPETA_SALIDA)
        b, ic, rs = res["base"], res["ic"], res["resumen"]
        print("--- Celdas válidas: %d (%.2f km2); bloques: %d" % (rs["n_validas"], rs["area_km2"], rs["n_bloques"]))
        print("--- % del área con pérdida de bosque por clase del IVT sin deforestación:")
        for k in (1, 2, 3):
            print("   %s: %s %% (IC95 %s a %s) ; razón de frecuencias %s" %
                  (NOMBRES_CLASE[k], _f(b["inc_cls%d" % k]), _f(ic["inc_cls%d" % k][0]), _f(ic["inc_cls%d" % k][1]),
                   _f(b["rf_cls%d" % k], 3)))
        print("--- % del área con pérdida de bosque según condición minera:")
        for k in (1, 2, 3):
            print("   %s: %s %% (IC95 %s a %s)" %
                  (NOMBRES_ESTATUS[k], _f(b["inc_est%d" % k]), _f(ic["inc_est%d" % k][0]), _f(ic["inc_est%d" % k][1])))
        print("   Diferencia título - libre: %s pp (IC95 %s a %s)" %
              (_f(b["dif_titulo_libre"]), _f(ic["dif_titulo_libre"][0]), _f(ic["dif_titulo_libre"][1])))
        print("--- AUC: %s (IC95 %s a %s)" % (_f(b["auc"], 3), _f(ic["auc"][0], 3), _f(ic["auc"][1], 3)))
        print("Tablas en: " + CARPETA_SALIDA)
    finally:
        arcpy.env.snapRaster, arcpy.env.extent, arcpy.env.cellSize, arcpy.env.mask = env_prev
        arcpy.CheckInExtension("Spatial")
