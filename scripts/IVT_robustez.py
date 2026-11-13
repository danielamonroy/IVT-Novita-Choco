import os
import csv
import numpy as np


VARIABLES = ["titulos", "solicitudes", "defor", "hidrico", "asent"]
W_BASE = {"titulos": 0.32, "solicitudes": 0.29, "defor": 0.16, "hidrico": 0.14, "asent": 0.09}


GDB_BASE = r"C:\ruta\IVT_base.gdb"
CARPETA_SALIDA = r"C:\ruta\robustez"
GDBS_ESCENARIOS = {
}
N_CLASES = 3
OAT_PASOS = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]
MC_N = 1000
MC_CONCENTRACION = 60
SEMILLA = 2026
MUESTRA_SPEARMAN = 200000


def escribir_csv(ruta, encabezado, filas):
    with open(ruta, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(encabezado)
        for f in filas:
            w.writerow(f)


def fmt(x, d=3):
    return ("%." + str(d) + "f") % x


def vector_pesos(w):
    v = np.array([w[k] for k in VARIABLES], dtype="float64")
    return v / v.sum()


def rangos(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spearman(a, b):
    ra, rb = rangos(a), rangos(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def jenks_cortes(valores, k=3, bins=512):
    v = valores[np.isfinite(valores)]
    cnt, edges = np.histogram(v, bins=bins)
    ok = cnt > 0
    c = cnt[ok].astype("float64")
    x = ((edges[:-1] + edges[1:]) / 2.0)[ok]
    hi = edges[1:][ok]
    n = len(c)
    if n < k:
        raise ValueError("Hay menos valores distintos que clases.")
    W = np.concatenate([[0.0], np.cumsum(c)])
    S = np.concatenate([[0.0], np.cumsum(c * x)])
    Q = np.concatenate([[0.0], np.cumsum(c * x * x)])
    D = np.full((k + 1, n + 1), np.inf)
    P = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for g in range(1, k + 1):
        for j in range(g, n + 1):
            i = np.arange(g - 1, j)
            w = W[j] - W[i]
            costo = (Q[j] - Q[i]) - (S[j] - S[i]) ** 2 / w
            tot = D[g - 1, i] + costo
            a = int(np.argmin(tot))
            D[g, j] = tot[a]
            P[g, j] = i[a]
    cortes = []
    j = n
    for g in range(k, 1, -1):
        i = P[g, j]
        cortes.append(hi[i - 1])
        j = i
    return sorted(cortes)


def clasificar(iv, cortes):
    cls = np.ones(iv.shape, dtype="int8")
    for c in cortes:
        cls += (iv > c).astype("int8")
    return cls


def matriz_confusion(a, b, k):
    return np.bincount((a.astype(int) - 1) * k + (b.astype(int) - 1), minlength=k * k).reshape(k, k)


def kappa_ponderado(cm):
    k = cm.shape[0]
    n = cm.sum()
    po_m = cm / n
    fila, col = po_m.sum(1), po_m.sum(0)
    i, j = np.indices((k, k))
    wt = 1.0 - np.abs(i - j) / (k - 1.0)
    po = (wt * po_m).sum()
    pe = (wt * np.outer(fila, col)).sum()
    return float((po - pe) / (1.0 - pe))


def comparar_clases(cls_ref, cls_x, k):
    cm = matriz_confusion(cls_ref, cls_x, k)
    a = cls_ref == k
    b = cls_x == k
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return {
        "pct_igual": float((cls_ref == cls_x).mean() * 100.0),
        "kappa_w": kappa_ponderado(cm),
        "pct_alta_ref": float(a.mean() * 100.0),
        "pct_alta_x": float(b.mean() * 100.0),
        "alta_retenida": inter / a.sum() * 100.0 if a.sum() else float("nan"),
        "jaccard_alta": inter / union if union else float("nan"),
        "cm": cm,
    }


def pesos_entropia(rec):
    x = rec.astype("float64")
    p = x / x.sum(0)
    e = -(p * np.log(p)).sum(0) / np.log(x.shape[0])
    d = 1.0 - e
    return d / d.sum()


def pesos_critic(rec):
    x = rec.astype("float64")
    rango = x.max(0) - x.min(0)
    rango[rango == 0] = 1.0
    z = (x - x.min(0)) / rango
    sd = z.std(0)
    r = np.corrcoef(z, rowvar=False)
    c = sd * (1.0 - r).sum(0)
    return c / c.sum()


def analizar(rec_dict, w_base, k, out_dir, cls_ref=None, mc_n=1000, conc=60, semilla=2026,
             pasos=(-0.30, -0.20, -0.10, 0.10, 0.20, 0.30), muestra=200000, trozo=25, log=print):
    os.makedirs(out_dir, exist_ok=True)
    rec_dict = {v: np.where(rec_dict[v] >= 1, rec_dict[v], np.nan) for v in VARIABLES}
    if cls_ref is not None:
        cls_ref = np.where(cls_ref >= 1, cls_ref, np.nan)
    forma = rec_dict[VARIABLES[0]].shape
    pila = np.stack([rec_dict[v].ravel() for v in VARIABLES], axis=1)
    valido = ~np.isnan(pila).any(axis=1)
    rec = pila[valido].astype("float32")
    N = rec.shape[0]
    log("Celdas válidas: %d" % N)
    rng = np.random.RandomState(semilla)
    idx = rng.choice(N, min(muestra, N), replace=False)

    w0 = vector_pesos(w_base)
    iv0 = rec @ w0.astype("float32")
    cortes0 = jenks_cortes(iv0, k)
    cls0 = clasificar(iv0, cortes0)
    log("Cortes Jenks base (propios): %s" % ", ".join(fmt(c) for c in cortes0))

    if cls_ref is not None:
        cr = cls_ref.ravel()[valido]
        ok = ~np.isnan(cr)
        ctrl = comparar_clases(cr[ok].astype("int8"), cls0[ok], k)
        log("Control Jenks propio vs ArcGIS: %.2f %% de celdas con la misma clase, kappa ponderado %.4f"
            % (ctrl["pct_igual"], ctrl["kappa_w"]))
        escribir_csv(os.path.join(out_dir, "robustez_0_control_jenks.csv"),
                     ["% celdas con la misma clase (Jenks propio vs ArcGIS)", "kappa ponderado"],
                     [[fmt(ctrl["pct_igual"], 2), fmt(ctrl["kappa_w"], 4)]])

    enc = (["escenario"] + ["peso_" + v for v in VARIABLES] +
           ["cortes (Jenks propio)", "% Baja", "% Media", "% Alta", "% Alta con cortes base",
            "% celdas misma clase que base", "kappa ponderado", "% de Alta base que sigue Alta",
            "Jaccard Alta", "Spearman IVT vs base"])

    def fila_escenario(nombre, w):
        iv = rec @ w.astype("float32")
        cs = jenks_cortes(iv, k)
        cl = clasificar(iv, cs)
        cmpar = comparar_clases(cls0, cl, k)
        pc = [float((cl == j).mean() * 100.0) for j in range(1, k + 1)]
        alta_cb = float((clasificar(iv, cortes0) == k).mean() * 100.0)
        rho = spearman(iv0[idx], iv[idx])
        return ([nombre] + [fmt(x, 3) for x in w] + [" ; ".join(fmt(c) for c in cs)] +
                [fmt(x, 2) for x in pc] + [fmt(alta_cb, 2), fmt(cmpar["pct_igual"], 2), fmt(cmpar["kappa_w"], 3),
                                          fmt(cmpar["alta_retenida"], 2), fmt(cmpar["jaccard_alta"], 3), fmt(rho, 3)])

    log("1/4 Pesos alternativos...")
    alt = [("AHP (base)", w0), ("Iguales", np.ones(len(VARIABLES)) / len(VARIABLES)),
           ("Entropía", pesos_entropia(rec[idx])), ("CRITIC", pesos_critic(rec[idx]))]
    filas = [fila_escenario(n, w) for n, w in alt]
    escribir_csv(os.path.join(out_dir, "robustez_1_pesos_alternativos.csv"), enc, filas)

    log("2/4 Sensibilidad de un factor a la vez (OAT)...")
    filas = []
    for i, v in enumerate(VARIABLES):
        for p in pasos:
            w = w0.copy()
            w[i] *= (1.0 + p)
            w = w / w.sum()
            filas.append(fila_escenario("%s %+d %%" % (v, int(round(p * 100))), w))
    escribir_csv(os.path.join(out_dir, "robustez_2_OAT.csv"), enc, filas)

    log("3/4 Monte Carlo (%d sorteos, Dirichlet, concentración %.0f)..." % (mc_n, conc))
    alfa = w0 * conc
    cuentas = np.zeros((N, k), dtype="uint16")
    alta_pct = []
    pesos_sorteados = []
    hechos = 0
    while hechos < mc_n:
        m = min(trozo, mc_n - hechos)
        Wm = rng.dirichlet(alfa, size=m)
        pesos_sorteados.append(Wm)
        iv = rec @ Wm.T.astype("float32")
        cls_m = np.ones(iv.shape, dtype="int8")
        for c in cortes0:
            cls_m += (iv > np.float32(c)).astype("int8")
        for j in range(1, k + 1):
            cuentas[:, j - 1] += (cls_m == j).sum(axis=1).astype("uint16")
        alta_pct.extend(((cls_m == k).mean(axis=0) * 100.0).tolist())
        hechos += m
    pesos_sorteados = np.vstack(pesos_sorteados)
    prob = cuentas / float(mc_n)
    p_alta = prob[:, k - 1]
    modal_igual = float((prob[np.arange(N), cls0.astype(int) - 1] >= 0.5).mean() * 100.0)
    estable = float((prob[np.arange(N), cls0.astype(int) - 1] >= 0.9).mean() * 100.0)
    a = np.array(alta_pct)
    resumen = [
        ["sorteos", str(mc_n)],
        ["concentración Dirichlet", fmt(conc, 0)],
        ["variación relativa media de los pesos sorteados (%)",
         " ; ".join("%s %.1f" % (v, 100 * pesos_sorteados[:, i].std() / pesos_sorteados[:, i].mean())
                    for i, v in enumerate(VARIABLES))],
        ["% área Alta (cortes base): media", fmt(a.mean(), 2)],
        ["% área Alta: percentil 2,5", fmt(np.percentile(a, 2.5), 2)],
        ["% área Alta: percentil 97,5", fmt(np.percentile(a, 97.5), 2)],
        ["% área Alta en la clasificación base", fmt((cls0 == k).mean() * 100.0, 2)],
        ["% celdas con P(Alta) >= 0,90", fmt((p_alta >= 0.9).mean() * 100.0, 2)],
        ["% celdas con P(Alta) <= 0,10", fmt((p_alta <= 0.1).mean() * 100.0, 2)],
        ["% celdas con 0,10 < P(Alta) < 0,90 (sensibles al peso)", fmt(((p_alta > 0.1) & (p_alta < 0.9)).mean() * 100.0, 2)],
        ["% celdas cuya clase base se repite en >= 50 % de los sorteos", fmt(modal_igual, 2)],
        ["% celdas cuya clase base se repite en >= 90 % de los sorteos", fmt(estable, 2)],
    ]
    escribir_csv(os.path.join(out_dir, "robustez_3_montecarlo.csv"), ["medida", "valor"], resumen)
    hist = np.histogram(p_alta, bins=np.linspace(0, 1, 11))[0] / float(N) * 100.0
    escribir_csv(os.path.join(out_dir, "robustez_3b_montecarlo_distribucion_P_alta.csv"),
                 ["intervalo de P(Alta)", "% de celdas"],
                 [["%.1f - %.1f" % (i / 10.0, (i + 1) / 10.0), fmt(h, 2)] for i, h in enumerate(hist)])

    prob_2d = np.full(pila.shape[0], np.nan, dtype="float32")
    prob_2d[valido] = p_alta
    log("4/4 Listo (la concordancia con otras geodatabases se hace en main()).")
    return {"valido": valido.reshape(forma), "prob_alta": prob_2d.reshape(forma),
            "cortes_base": cortes0, "clase_base_propia": cls0}


def concordancia_gdbs(cls_base, otras, k, out_dir, log=print):
    filas = []
    cms = []
    cls_base = np.where(cls_base >= 1, cls_base, np.nan)
    for nombre, cl in otras.items():
        cl = np.where(cl >= 1, cl, np.nan)
        if cl.shape != cls_base.shape:
            log("AVISO: '%s' tiene otra cuadrícula (%s vs %s); no se compara." % (nombre, cl.shape, cls_base.shape))
            continue
        ok = ~np.isnan(cls_base) & ~np.isnan(cl)
        a, b = cls_base[ok].astype("int8"), cl[ok].astype("int8")
        r = comparar_clases(a, b, k)
        filas.append([nombre, str(int(ok.sum())), fmt(r["pct_igual"], 2), fmt(r["kappa_w"], 3),
                      fmt(r["pct_alta_ref"], 2), fmt(r["pct_alta_x"], 2), fmt(r["alta_retenida"], 2),
                      fmt(r["jaccard_alta"], 3)])
        cm_pct = r["cm"] / r["cm"].sum(1, keepdims=True) * 100.0
        for i in range(k):
            cms.append([nombre, "base clase %d" % (i + 1)] + [fmt(x, 2) for x in cm_pct[i]])
    escribir_csv(os.path.join(out_dir, "robustez_4_concordancia_escenarios.csv"),
                 ["escenario", "celdas comparadas", "% celdas misma clase", "kappa ponderado", "% Alta base",
                  "% Alta escenario", "% de Alta base que sigue Alta", "Jaccard Alta"], filas)
    escribir_csv(os.path.join(out_dir, "robustez_4b_matriz_confusion_pct_por_fila.csv"),
                 ["escenario", "fila"] + ["se clasifica %d" % (j + 1) for j in range(k)], cms)
    return filas


def descriptivos_y_correlacion(dist_dict, rec_dict, out_dir, muestra=300000, semilla=2026):
    os.makedirs(out_dir, exist_ok=True)
    rec = {v: np.where(a >= 1, a, np.nan) for v, a in rec_dict.items()}
    filas = []
    for k, a in dist_dict.items():
        x = a[~np.isnan(a)]
        filas.append([k, str(x.size), fmt(x.min(), 2), fmt(x.max(), 2), fmt(np.median(x), 2), fmt(x.mean(), 2),
                      fmt((x == 0).mean() * 100.0, 2)])
    escribir_csv(os.path.join(out_dir, "robustez_0b_descriptivos_distancias.csv"),
                 ["variable (distancia en m)", "celdas del municipio", "minimo", "maximo", "mediana", "media",
                  "% celdas con distancia 0 (sobre el elemento)"], filas)
    filas = []
    for k, a in rec.items():
        x = a[~np.isnan(a)]
        filas.append([k, str(x.size)] + [fmt((x == j).mean() * 100.0, 2) for j in range(1, 6)])
    escribir_csv(os.path.join(out_dir, "robustez_0c_distribucion_clases_1a5.csv"),
                 ["criterio", "celdas del municipio"] + ["%% clase %d" % j for j in range(1, 6)], filas)
    nombres = list(rec.keys())
    pila = np.stack([rec[k].ravel() for k in nombres], axis=1)
    pila = pila[~np.isnan(pila).any(axis=1)]
    rng = np.random.RandomState(semilla)
    if pila.shape[0] > muestra:
        pila = pila[rng.choice(pila.shape[0], muestra, replace=False)]
    corr = np.corrcoef(np.stack([rangos(pila[:, j]) for j in range(pila.shape[1])], axis=1), rowvar=False)
    escribir_csv(os.path.join(out_dir, "robustez_0d_correlacion_spearman.csv"), [""] + nombres,
                 [[k] + [fmt(x, 3) for x in corr[i]] for i, k in enumerate(nombres)])
    return corr


def leer_raster_nan(arcpy, path):
    import arcpy.sa as sa
    r = arcpy.Raster(path)
    a = arcpy.RasterToNumPyArray(r).astype("float64")
    nulo = arcpy.RasterToNumPyArray(sa.IsNull(r)).astype(bool)
    a[nulo] = np.nan
    return a


def areas_por_clase(clases, k, area_celda_m2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filas = []
    for nombre, cl in clases.items():
        cl = np.where(cl >= 1, cl, np.nan)
        n = [int((cl == j).sum()) for j in range(1, k + 1)]
        tot = float(sum(n))
        filas.append([nombre, str(int(tot))] + [fmt(x * area_celda_m2 / 1e6, 2) for x in n] +
                     [fmt(x / tot * 100.0, 2) for x in n])
    escribir_csv(os.path.join(out_dir, "robustez_4c_areas_por_clase.csv"),
                 ["escenario", "celdas"] + ["km2 clase %d" % j for j in range(1, k + 1)] +
                 ["%% clase %d" % j for j in range(1, k + 1)], filas)
    return filas


def main_concordancia():
    import arcpy
    arcpy.CheckOutExtension("Spatial")
    try:
        ruta_base = os.path.join(GDB_BASE, "IVT_jenks")
        r = arcpy.Raster(ruta_base)
        area_celda = r.meanCellWidth * r.meanCellHeight
        cls_base = leer_raster_nan(arcpy, ruta_base)
        otras = {}
        for nombre, gdb in GDBS_ESCENARIOS.items():
            try:
                otras[nombre] = leer_raster_nan(arcpy, os.path.join(gdb, "IVT_jenks"))
            except Exception as e:
                print("AVISO: no se pudo leer '%s' (%s): %s" % (nombre, gdb, e))
        todo = {"Base": cls_base}
        todo.update(otras)
        print("--- Área por clase (km2 baja/media/alta ; % baja/media/alta) ---")
        for f in areas_por_clase(todo, N_CLASES, area_celda, CARPETA_SALIDA):
            print("%s: %s / %s / %s km2 ; %s / %s / %s %%" % (f[0], f[2], f[3], f[4], f[5], f[6], f[7]))
        print("--- Concordancia con la base (celda a celda) ---")
        for f in concordancia_gdbs(cls_base, otras, N_CLASES, CARPETA_SALIDA, log=print):
            print("%s: %s %% misma clase ; kappa ponderado %s ; %s %% de la clase Alta base sigue Alta ; Jaccard Alta %s"
                  % (f[0], f[2], f[3], f[6], f[7]))
        print("Tablas en: " + CARPETA_SALIDA)
    finally:
        arcpy.CheckInExtension("Spatial")


def main():
    import arcpy
    arcpy.CheckOutExtension("Spatial")

    def L(m):
        print(m)

    try:
        rec = {v: leer_raster_nan(arcpy, os.path.join(GDB_BASE, "rec_" + v)) for v in VARIABLES}
        cls_ref = leer_raster_nan(arcpy, os.path.join(GDB_BASE, "IVT_jenks"))
        forma = rec[VARIABLES[0]].shape
        if any(a.shape != forma for a in rec.values()):
            raise RuntimeError("Los ráster rec_* no tienen la misma cuadrícula: %s" % {k: a.shape for k, a in rec.items()})
        if cls_ref.shape != forma:
            L("AVISO: IVT_jenks tiene otra cuadrícula (%s vs %s); se omite el control contra ArcGIS." % (cls_ref.shape, forma))
            cls_ref = None
        res = analizar(rec, W_BASE, N_CLASES, CARPETA_SALIDA, cls_ref=cls_ref, mc_n=MC_N, conc=MC_CONCENTRACION,
                       semilla=SEMILLA, pasos=OAT_PASOS, muestra=MUESTRA_SPEARMAN, log=L)

        try:
            dist = {k: leer_raster_nan(arcpy, os.path.join(GDB_BASE, "dist_" + k))
                    for k in ["titulos", "solicitudes", "hidrico", "asent"]}
            descriptivos_y_correlacion(dist, rec, CARPETA_SALIDA, semilla=SEMILLA)
            L("Descriptivos y correlación de Spearman guardados (robustez_0b, 0c y 0d).")
        except Exception as e:
            L("No se pudieron calcular los descriptivos (lo demás sí quedó): %s" % e)

        try:
            base = arcpy.Raster(os.path.join(GDB_BASE, "rec_titulos"))
            arr = np.where(np.isnan(res["prob_alta"]), -9999.0, res["prob_alta"]).astype("float32")
            r = arcpy.NumPyArrayToRaster(arr, arcpy.Point(base.extent.XMin, base.extent.YMin),
                                         base.meanCellWidth, base.meanCellHeight, -9999)
            salida = os.path.join(GDB_BASE, "MC_prob_alta")
            r.save(salida)
            arcpy.management.DefineProjection(salida, base.spatialReference)
            L("Ráster de probabilidad guardado: " + salida)
        except Exception as e:
            L("No se pudo guardar el ráster de probabilidad (las tablas sí quedaron): %s" % e)

        if GDBS_ESCENARIOS:
            otras = {n: leer_raster_nan(arcpy, os.path.join(g, "IVT_jenks")) for n, g in GDBS_ESCENARIOS.items()}
            concordancia_gdbs(cls_ref if cls_ref is not None else leer_raster_nan(arcpy, os.path.join(GDB_BASE, "IVT_jenks")),
                              otras, N_CLASES, CARPETA_SALIDA, log=L)
        L("Tablas en: " + CARPETA_SALIDA)
    finally:
        arcpy.CheckInExtension("Spatial")


if __name__ == "__main__" and "__file__" in globals():
    main()
