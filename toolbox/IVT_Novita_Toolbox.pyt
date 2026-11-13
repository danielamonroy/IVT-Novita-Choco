import os
import csv
import arcpy


VARIABLES = ["titulos", "solicitudes", "defor", "hidrico", "asent"]
MATRIZ_AHP = [
    [1.0, 1.0, 2.0, 3.0, 3.0],
    [1.0, 1.0, 2.0, 2.0, 3.0],
    [1 / 2., 1 / 2., 1.0, 1.0, 2.0],
    [1 / 3., 1 / 2., 1.0, 1.0, 2.0],
    [1 / 3., 1 / 3., 1 / 2., 1 / 2., 1.0],
]
RI_SAATY = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
PESOS_DEFECTO = {"titulos": 0.32, "solicitudes": 0.29, "defor": 0.16, "hidrico": 0.14, "asent": 0.09}


def pesos_ahp(matriz):
    n = len(matriz)
    col = [sum(matriz[i][j] for i in range(n)) for j in range(n)]
    w = [sum(matriz[i][j] / col[j] for j in range(n)) / n for i in range(n)]
    lam = sum(sum(matriz[i][j] * w[j] for j in range(n)) / w[i] for i in range(n)) / n
    ci = (lam - n) / (n - 1)
    cr = ci / RI_SAATY[n] if RI_SAATY[n] else 0.0
    return w, lam, ci, cr


def aplicar_exclusion(pesos, excluida):
    if not excluida or excluida == "Ninguna":
        return dict(pesos)
    if excluida not in pesos:
        raise ValueError("Variable a excluir desconocida: %s" % excluida)
    resto = sum(x for k, x in pesos.items() if k != excluida)
    return {k: (0.0 if k == excluida else x / resto) for k, x in pesos.items()}


def _rank_avg(x):
    import numpy as np
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def leer_raster_nan(path):
    import numpy as np
    import arcpy.sa as sa
    r = arcpy.Raster(path)
    a = arcpy.RasterToNumPyArray(r).astype("float64")
    nulo = arcpy.RasterToNumPyArray(sa.IsNull(r)).astype(bool)
    a[nulo] = np.nan
    return a


def estadisticos_y_correlacion(dist_arrays, rec_arrays, out_folder, muestra=300000, semilla=2026):
    import numpy as np
    filas = []
    for k, a in dist_arrays.items():
        v = a[~np.isnan(a)]
        filas.append([k, int(v.size), float(v.min()), float(v.max()), float(np.median(v)), float(v.mean()),
                      float((v == 0).sum()) / v.size * 100.0])
    f1 = os.path.join(out_folder, "IVT_estadisticos_variables.csv")
    with open(f1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["variable (distancia en m)", "celdas", "minimo", "maximo", "mediana", "media", "% celdas con distancia 0"])
        for r in filas:
            w.writerow([r[0], r[1]] + ["%.2f" % x for x in r[2:]])
    nombres = list(rec_arrays.keys())
    stack = np.vstack([rec_arrays[k].ravel() for k in nombres]).T
    ok = ~np.isnan(stack).any(axis=1)
    stack = stack[ok]
    rng = np.random.RandomState(semilla)
    if stack.shape[0] > muestra:
        stack = stack[rng.choice(stack.shape[0], muestra, replace=False)]
    rangos = np.vstack([_rank_avg(stack[:, j]) for j in range(stack.shape[1])]).T
    corr = np.corrcoef(rangos, rowvar=False)
    f2 = os.path.join(out_folder, "IVT_correlacion_spearman.csv")
    with open(f2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow([""] + nombres)
        for i, k in enumerate(nombres):
            w.writerow([k] + ["%.3f" % x for x in corr[i]])
    return f1, f2


class Toolbox(object):
    def __init__(self):
        self.label = "Toolbox IVT Nóvita"
        self.alias = "IVT_Novita"
        self.tools = [CalcularIVTCompleto]


class CalcularIVTCompleto(object):
    def __init__(self):
        self.label = "IVT Nóvita: cálculo completo (IVT, Jenks, áreas)"
        self.description = (
            "Calcula el IVT (5 variables) con álgebra de mapas ponderada (AHP), lo clasifica por Jenks, "
            "calcula áreas, tabula el área por clase en cada consejo comunitario y exporta estadísticos."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        P = arcpy.Parameter
        ps = []

        def add(name, label, dtype, req="Required", direction="Input", default=None, filt=None, cat=None):
            p = P(displayName=label, name=name, datatype=dtype, parameterType=req, direction=direction)
            if default is not None:
                p.value = default
            if filt is not None:
                p.filter.type = "ValueList"
                p.filter.list = filt
            if cat:
                p.category = cat
            ps.append(p)
            return p

        add("in_limite", "Límite municipal de Nóvita (máscara)", "GPFeatureLayer")
        add("in_titulos", "Títulos mineros vigentes", "GPFeatureLayer")
        add("in_solicitudes", "Solicitudes mineras vigentes", "GPFeatureLayer")
        add("in_dren_sencillo", "Drenaje sencillo", "GPFeatureLayer")
        add("in_dren_doble", "Drenaje doble", "GPFeatureLayer")
        add("in_worldpop", "WorldPop población 2025, 100 m (personas por celda; asentamientos)", "GPRasterLayer")
        add("in_hansen", "Hansen GFC lossyear (GeoTIFF, valores 0-25)", "GPRasterLayer")
        add("in_treecover", "Hansen GFC treecover2000 (cobertura arbórea año 2000, 0-100 %)", "GPRasterLayer")
        add("in_datamask", "Hansen GFC datamask (opcional; 1 = tierra, 2 = agua permanente)", "GPRasterLayer",
            req="Optional")
        add("in_consejos", "Consejos comunitarios (unidad de reporte; opcional)", "GPFeatureLayer", req="Optional")
        add("out_folder", "Carpeta de salida (se crea la GDB dentro)", "DEFolder")
        add("nombre_gdb", "Nombre de la geodatabase de salida", "GPString",
            req="Optional", default="IVT_Novita", cat="Opciones del modelo")

        add("cell_size", "Tamaño de celda (m)", "GPDouble", req="Optional", default=30.0, cat="Opciones del modelo")
        add("metodo_rec", "Método de reclasificación 1-5 (EQUAL_AREA = cuantiles; EQUAL_INTERVAL = intervalos iguales)",
            "GPString", req="Optional", default="EQUAL_AREA", filt=["EQUAL_AREA", "EQUAL_INTERVAL"], cat="Opciones del modelo")
        add("buffer_km", "Margen de recorte alrededor del municipio (km)", "GPDouble", req="Optional", default=10.0,
            cat="Opciones del modelo")
        add("n_clases", "Número de clases del IVT (Jenks)", "GPLong", req="Optional", default=3, filt=[3, 4, 5],
            cat="Opciones del modelo")
        add("canopy_min", "Deforestación: cobertura arbórea mínima en 2000 para considerar bosque (%)", "GPLong",
            req="Optional", default=30, cat="Opciones del modelo")
        add("anio_min", "Deforestación: año mínimo de pérdida", "GPLong", req="Optional",
            default=2001, cat="Opciones del modelo")
        add("radio_defor", "Deforestación: radio de vecindad para la densidad (m)", "GPDouble", req="Optional",
            default=500.0, cat="Opciones del modelo")
        add("umbral_pob", "WorldPop: una celda cuenta como asentamiento si su población es mayor que (personas/celda)",
            "GPDouble", req="Optional", default=0.0, cat="Opciones del modelo")

        add("w_titulos", "Peso títulos mineros", "GPDouble", req="Optional", default=PESOS_DEFECTO["titulos"], cat="Pesos AHP")
        add("w_solicitudes", "Peso solicitudes mineras", "GPDouble", req="Optional", default=PESOS_DEFECTO["solicitudes"], cat="Pesos AHP")
        add("w_defor", "Peso deforestación", "GPDouble", req="Optional", default=PESOS_DEFECTO["defor"], cat="Pesos AHP")
        add("w_hidrico", "Peso red hídrica", "GPDouble", req="Optional", default=PESOS_DEFECTO["hidrico"], cat="Pesos AHP")
        add("w_asent", "Peso asentamientos (celdas con población)", "GPDouble", req="Optional",
            default=PESOS_DEFECTO["asent"], cat="Pesos AHP")
        add("variable_excluida", "Sensibilidad: excluir una variable y reescalar los demás pesos a 1", "GPString",
            req="Optional", default="Ninguna", filt=["Ninguna"] + VARIABLES, cat="Pesos AHP")

        add("campo_consejo", "Campo con el nombre del consejo comunitario (para área por consejo)", "GPString",
            req="Optional", cat="Unidad de reporte")
        add("exportar_stats", "Exportar estadísticos descriptivos y correlaciones (CSV)", "GPBoolean", req="Optional",
            default=True, cat="Opciones del modelo")
        return ps

    def isLicensed(self):
        return arcpy.CheckExtension("Spatial") == "Available"

    def updateMessages(self, parameters):
        nom = ["w_" + k for k in VARIABLES]
        pm = {p.name: p for p in parameters}
        ws = [pm[n].value for n in nom]
        if all(v is not None for v in ws) and abs(sum(float(v) for v in ws) - 1.0) > 0.011:
            pm["w_titulos"].setWarningMessage("Los pesos deben sumar 1,00 (suman %.3f)." % sum(float(v) for v in ws))
        return

    def execute(self, parameters, messages):
        import arcpy.sa as sa
        v = {p.name: p for p in parameters}

        def val(n):
            return v[n].valueAsText

        def num(n, d):
            t = v[n].valueAsText
            return float(t.replace(",", ".")) if t else d

        spatial = False
        try:
            arcpy.CheckOutExtension("Spatial")
            spatial = True
            ctm12 = arcpy.SpatialReference(9377)

            cell = num("cell_size", 30.0)
            pesos = {k: num("w_" + k, PESOS_DEFECTO[k]) for k in VARIABLES}
            if abs(sum(pesos.values()) - 1.0) > 0.011:
                raise ValueError("Los pesos suman %.3f; deben sumar 1,00." % sum(pesos.values()))
            excluida = val("variable_excluida") or "Ninguna"
            if excluida != "Ninguna":
                pesos = aplicar_exclusion(pesos, excluida)
                messages.addWarningMessage("Escenario SIN la variable '%s': pesos reescalados a 1." % excluida)
            n_clases = int(num("n_clases", 3))

            w, lam, ci, cr = pesos_ahp(MATRIZ_AHP)
            messages.addMessage("Matriz AHP (5x5): lambda_max=%.3f  CI=%.4f  CR=%.4f (%s)" %
                                (lam, ci, cr, "consistente" if cr < 0.10 else "no consistente"))
            messages.addMessage("Pesos calculados de la matriz (%s): %s" %
                                (", ".join(VARIABLES), ", ".join("%.3f" % x for x in w)))
            messages.addMessage("Pesos usados: " + str(pesos))

            nombre = (val("nombre_gdb") or "IVT_Novita").replace(".gdb", "")
            out_gdb = os.path.join(val("out_folder"), nombre + ".gdb")
            if not arcpy.Exists(out_gdb):
                arcpy.management.CreateFileGDB(os.path.dirname(out_gdb), os.path.basename(out_gdb))
            arcpy.env.workspace = out_gdb
            arcpy.env.scratchWorkspace = out_gdb
            arcpy.env.overwriteOutput = True
            arcpy.env.outputCoordinateSystem = ctm12

            def G(name):
                return os.path.join(out_gdb, name)

            def to_ctm12(src, name, clip=None):
                sr = arcpy.Describe(src).spatialReference
                cur = src
                if sr.factoryCode != 9377:
                    cur = G(name + "_prj")
                    arcpy.management.Project(src, cur, ctm12)
                if clip:
                    out = G(name)
                    arcpy.analysis.Clip(cur, clip, out)
                    return out
                if cur == src:
                    out = G(name)
                    arcpy.management.CopyFeatures(src, out)
                    return out
                return cur

            messages.addMessage(">>> A. Reproyectando y recortando a CTM12...")
            limite = to_ctm12(val("in_limite"), "limite_ctm12")
            margen = G("limite_margen")
            buf_km = num("buffer_km", 10.0)
            arcpy.analysis.Buffer(limite, margen, "%s Meters" % (buf_km * 1000.0))
            arcpy.env.extent = margen
            arcpy.env.mask = limite
            arcpy.env.cellSize = cell

            titulos = to_ctm12(val("in_titulos"), "titulos_ctm12", margen)
            solicitudes = to_ctm12(val("in_solicitudes"), "solicitudes_ctm12", margen)
            dsen = to_ctm12(val("in_dren_sencillo"), "dren_sencillo_ctm12", margen)
            ddob = to_ctm12(val("in_dren_doble"), "dren_doble_ctm12", margen)
            narp = to_ctm12(val("in_consejos"), "consejos_ctm12", margen) if val("in_consejos") else None

            if arcpy.Describe(ddob).shapeType == "Polygon":
                ddob_l = G("dren_doble_linea")
                arcpy.management.PolygonToLine(ddob, ddob_l)
                ddob = ddob_l
            drenajes = G("drenajes_merged")
            arcpy.management.Merge([dsen, ddob], drenajes)

            def recortar_raster(path, name):
                sr = arcpy.Describe(path).spatialReference
                e = arcpy.Describe(margen).extent.projectAs(sr)
                rect = "%s %s %s %s" % (e.XMin, e.YMin, e.XMax, e.YMax)
                out = G(name)
                arcpy.management.Clip(path, rect, out, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
                return out

            messages.addMessage(">>> B. Distancias euclidianas...")
            dist = {}
            for k, fc in [("titulos", titulos), ("solicitudes", solicitudes), ("hidrico", drenajes)]:
                r = sa.EucDistance(fc, cell_size=cell)
                r.save(G("dist_" + k))
                dist[k] = r
            umbral = num("umbral_pob", 0.0)
            messages.addMessage("   Asentamientos: celdas WorldPop con más de %.2f personas." % umbral)
            wp_clip = recortar_raster(val("in_worldpop"), "worldpop_clip")
            m_prev = arcpy.env.mask
            arcpy.env.mask = None
            wp_prj = G("worldpop_prj")
            arcpy.management.ProjectRaster(wp_clip, wp_prj, ctm12, "NEAREST", "%s %s" % (cell, cell))
            wp_src = sa.Con(sa.Raster(wp_prj) > umbral, 1)
            wp_src.save(G("worldpop_fuente"))
            d_wp = sa.EucDistance(wp_src, cell_size=cell)
            arcpy.env.mask = m_prev
            d_as = sa.ExtractByMask(d_wp, limite)
            d_as.save(G("dist_asent"))
            dist["asent"] = d_as

            metodo = val("metodo_rec") or "EQUAL_AREA"
            messages.addMessage(">>> C. Reclasificando 1-5 (%s, invertido)..." % metodo)
            rec = {}
            for k, r in dist.items():
                rr = 6 - sa.Slice(r, 5, metodo)
                rr.save(G("rec_" + k))
                rec[k] = rr

            anio_min = int(num("anio_min", 2001))
            radio = num("radio_defor", 500.0)
            messages.addMessage(">>> D. Deforestación Hansen: pérdida de bosque (>=%d %% en 2000) desde %d, densidad en radio %.0f m..." %
                                (int(num("canopy_min", 30)), anio_min, radio))
            canopy = int(num("canopy_min", 30))

            recortar_hansen = recortar_raster

            h_loss = recortar_hansen(val("in_hansen"), "hansen_lossyear_clip")
            h_tree = recortar_hansen(val("in_treecover"), "hansen_treecover_clip")
            h_mask = recortar_hansen(val("in_datamask"), "hansen_datamask_clip") if val("in_datamask") else None

            env_prev = (arcpy.env.extent, arcpy.env.mask, arcpy.env.outputCoordinateSystem, arcpy.env.cellSize)
            arcpy.env.extent = None
            arcpy.env.mask = None
            arcpy.env.outputCoordinateSystem = None
            arcpy.env.cellSize = None
            lossyear = sa.Raster(h_loss)
            bosque2000 = sa.Raster(h_tree) >= canopy
            perdida = sa.Con((lossyear >= (anio_min - 2000)) & bosque2000, 1, 0)
            if h_mask:
                perdida = sa.Con(sa.Raster(h_mask) == 1, perdida)
            perdida_geo = G("perdida_geo")
            perdida.save(perdida_geo)
            arcpy.env.extent, arcpy.env.mask, arcpy.env.outputCoordinateSystem, arcpy.env.cellSize = env_prev
            perdida_prj = G("perdida_prj")
            arcpy.management.ProjectRaster(perdida_geo, perdida_prj, ctm12, "NEAREST", "%s %s" % (cell, cell))
            perdida_m = sa.ExtractByMask(perdida_prj, margen)
            dens = sa.FocalStatistics(perdida_m, sa.NbrCircle(radio, "MAP"), "MEAN", "DATA")
            dens = sa.ExtractByMask(dens, limite)
            dens.save(G("defor_densidad"))
            pos = sa.Con(dens > 0, dens)
            rec_defor = sa.Con(dens > 0, sa.Slice(pos, 4, "EQUAL_AREA") + 1, 1)
            rec_defor.save(G("rec_defor"))

            messages.addMessage(">>> E. Álgebra de mapas (IVT, 5 variables)...")
            ivt = (rec["titulos"] * pesos["titulos"] + rec["solicitudes"] * pesos["solicitudes"] +
                   rec_defor * pesos["defor"] + rec["hidrico"] * pesos["hidrico"] +
                   rec["asent"] * pesos["asent"])
            ivt.save(G("IVT"))

            messages.addMessage(">>> F. Clasificación Jenks (%d clases) y áreas..." % n_clases)
            ivt_cls = sa.Slice(ivt, n_clases, "NATURAL_BREAKS")
            ivt_cls.save(G("IVT_jenks"))

            tab_cortes = G("IVT_cortes_jenks")
            sa.ZonalStatisticsAsTable(ivt_cls, "Value", ivt, tab_cortes, "DATA", "MIN_MAX_MEAN")
            if n_clases == 3:
                nombres = {1: "Vulnerabilidad Baja", 2: "Vulnerabilidad Media", 3: "Vulnerabilidad Alta"}
            else:
                nombres = {i: "Clase %d de %d (1 = más baja)" % (i, n_clases) for i in range(1, n_clases + 1)}
            for r in arcpy.da.SearchCursor(tab_cortes, ["VALUE", "MIN", "MAX", "MEAN"]):
                messages.addMessage("   %s: IVT %.3f - %.3f (media %.3f)" % (nombres[r[0]], r[1], r[2], r[3]))

            poly = G("IVT_poly_tmp")
            arcpy.conversion.RasterToPolygon(ivt_cls, poly, "NO_SIMPLIFY", "VALUE")
            out_vec = G("IVT_jenks_vector")
            arcpy.management.Dissolve(poly, out_vec, "gridcode")
            arcpy.management.AddField(out_vec, "CATEGORIA", "TEXT", field_length=40)
            arcpy.management.AddField(out_vec, "AREA_HA", "DOUBLE")
            arcpy.management.AddField(out_vec, "AREA_KM2", "DOUBLE")
            arcpy.management.AddField(out_vec, "PORC", "DOUBLE")
            with arcpy.da.UpdateCursor(out_vec, ["gridcode", "CATEGORIA", "SHAPE@"]) as cur:
                for row in cur:
                    row[1] = nombres[row[0]]
                    cur.updateRow(row)
            total = sum(r[0] for r in arcpy.da.SearchCursor(out_vec, ["SHAPE@AREA"]))
            with arcpy.da.UpdateCursor(out_vec, ["SHAPE@AREA", "AREA_HA", "AREA_KM2", "PORC", "CATEGORIA"]) as cur:
                for row in cur:
                    row[1] = row[0] / 10000.0
                    row[2] = row[0] / 1e6
                    row[3] = row[0] / total * 100.0
                    cur.updateRow(row)
                    messages.addMessage("   %s: %.2f km2 (%.2f %%)" % (row[4], row[2], row[3]))
            messages.addMessage("   Área total: %.2f km2" % (total / 1e6))

            campo = val("campo_consejo")
            if campo and narp:
                messages.addMessage(">>> G. Área por clase en cada consejo comunitario...")
                tab_cons = G("area_clase_por_consejo")
                sa.TabulateArea(narp, campo, ivt_cls, "Value", tab_cons, cell)
                messages.addMessage("   Tabla: %s (áreas en m2; columnas VALUE_1 = clase más baja ... VALUE_%d = más alta)" %
                                    (tab_cons, n_clases))

            if v["exportar_stats"].value:
                messages.addMessage(">>> H. Estadísticos descriptivos y correlación de Spearman...")
                try:
                    import numpy as np
                    arcpy.env.extent = None
                    arcpy.env.mask = None
                    dist_arr = {k: leer_raster_nan(G("dist_" + k))
                                for k in ["titulos", "solicitudes", "hidrico", "asent"]}
                    rec_arr = {k: leer_raster_nan(G("rec_" + k))
                               for k in ["titulos", "solicitudes", "defor", "hidrico", "asent"]}
                    for a_ in rec_arr.values():
                        a_[a_ < 1] = np.nan
                    f1, f2 = estadisticos_y_correlacion(dist_arr, rec_arr, val("out_folder"))
                    messages.addMessage("   CSV: %s ; %s" % (f1, f2))
                except Exception as e2:
                    messages.addWarningMessage("No se pudieron exportar los estadísticos (el resto del proceso sí terminó): %s" % e2)

            messages.addMessage(">>> PROCESO CONCLUIDO. Salidas en: " + out_gdb)

        except arcpy.ExecuteError:
            arcpy.AddError(arcpy.GetMessages(2))
        except Exception as e:
            arcpy.AddError("Error: %s" % e)
        finally:
            if spatial:
                arcpy.CheckInExtension("Spatial")
