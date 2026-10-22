import csv
import io
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson, norm
from scipy.optimize import minimize
from difflib import get_close_matches

# =====================================================================
# CONFIGURACIÓN: Pega aquí tu API Key gratuita de football-data.org
# =====================================================================
API_KEY = "6ff62db217f54b688146de5251f835af"

LIGAS_DISPONIBLES = {
    "⚽ UEFA Champions League": ("CL", "SOCCER"),
    "⚽ LaLiga España": ("PD", "SOCCER"),
    "⚽ Premier League Inglaterra": ("PL", "SOCCER"),
    "⚽ Serie A Italia": ("SA", "SOCCER"),
    "⚽ Bundesliga Alemania": ("BL1", "SOCCER"),
    "⚽ Ligue 1 Francia": ("FL1", "SOCCER"),
    "⚽ MLS Estados Unidos": ("MLS", "SOCCER"),
    "🏀 NBA (Básquetbol EE.UU.)": ("NBA", "BASKETBALL"),
    "🏈 NFL (Fútbol Americano)": ("NFL", "FOOTBALL"),
}

CSV_LIGAS_EUROPEAS = {
    "PD": "SP1", "PL": "E0", "SA": "I1", "BL1": "D1", "FL1": "F1",
    "PPL": "P1", "DED": "N1", "BEL": "B1", "SCO": "SC0", "TUR": "T1", "GRE": "G1"
}

CATALOGO_MERCADOS_GOLES = [
    ("Victoria Local (1)", lambda i, j: i > j),
    ("Empate (X)", lambda i, j: i == j),
    ("Victoria Visita (2)", lambda i, j: i < j),
    ("Doble Oportunidad Local/Empate (1X)", lambda i, j: i >= j),
    ("Doble Oportunidad Empate/Visita (X2)", lambda i, j: i <= j),
    ("Más de 1.5 Goles (Over 1.5)", lambda i, j: (i + j) >= 2),
    ("Más de 2.5 Goles (Over 2.5)", lambda i, j: (i + j) >= 3),
    ("Menos de 2.5 Goles (Under 2.5)", lambda i, j: (i + j) <= 2),
    ("Más de 3.5 Goles (Over 3.5)", lambda i, j: (i + j) >= 4),
    ("Ambos Equipos Anotan: Sí (BTTS)", lambda i, j: i >= 1 and j >= 1),
    ("Ambos Equipos Anotan: No", lambda i, j: i == 0 or j == 0),
    ("Local Anota Más de 1.5 Goles", lambda i, j: i >= 2),
    ("Visita Anota Más de 1.5 Goles", lambda i, j: j >= 2),
]

st.set_page_config(page_title="QuantBet Pro V8", page_icon="🏆", layout="wide")


# =====================================================================
# COMPONENTE VISUAL DE TARJETAS SIN CORTE DE TEXTO (...)
# =====================================================================
def mostrar_tarjeta_quant(columna, titulo, valor_principal, subtexto_verde):
    """Renderiza una tarjeta idéntica a st.metric pero 100% libre de cortes (...)"""
    html_card = f"""
    <div style="background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; min-height: 118px;">
        <div style="font-size: 0.88rem; color: rgba(250, 250, 250, 0.75); margin-bottom: 6px; font-weight: 500;">
            {titulo}
        </div>
        <div style="font-size: 1.65rem; font-weight: 700; color: #FFFFFF; line-height: 1.2;
                    margin-bottom: 10px; word-wrap: break-word; white-space: normal;">
            {valor_principal}
        </div>
        <div style="display: inline-block; background-color: rgba(33, 195, 84, 0.16); color: #3dd56d;
                    padding: 4px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 500;
                    line-height: 1.3; word-wrap: break-word; white-space: normal;">
            ↑ {subtexto_verde}
        </div>
    </div>
    """
    columna.markdown(html_card, unsafe_allow_html=True)


def nombre_corto_us(nombre_completo):
    partes = str(nombre_completo).strip().split()
    if len(partes) >= 2 and partes[-2] in ("Trail", "Red", "Blue", "White", "Golden"):
        return " ".join(partes[-2:])
    return partes[-1] if partes else nombre_completo


def parse_win_pct(rec_texto, base_pct):
    try:
        partes = [int(x) for x in str(rec_texto).split("-")[:2]]
        if sum(partes) > 0:
            return partes[0] / sum(partes)
    except Exception:
        pass
    return base_pct


@st.cache_data(ttl=1800)
def cargar_datos_nba_nfl_espn(deporte_cod):
    ruta = "basketball/nba" if deporte_cod == "NBA" else "football/nfl"
    pts_default = 114.2 if deporte_cod == "NBA" else 22.8
    equipos_db = {}

    # Si la temporada actual acaba de empezar (0-0), consulta la temporada previa como base real
    for season_param in ["", "?season=2026", "?season=2025"]:
        url_st = f"https://site.api.espn.com/apis/v2/sports/{ruta}/standings{season_param}"
        try:
            r = requests.get(url_st, timeout=12)
            if r.status_code == 200:
                data = r.json()
                equipos_temp = {}
                partidos_max = 0
                for conf in data.get("children", []):
                    entries = list(conf.get("standings", {}).get("entries", []))
                    for sub in conf.get("children", []):
                        entries.extend(sub.get("standings", {}).get("entries", []))
                    for ent in entries:
                        team_info = ent.get("team", {})
                        nombre = team_info.get("displayName")
                        if not nombre:
                            continue
                        stats_dict, stats_str = {}, {}
                        for s in ent.get("stats", []):
                            sn = s.get("name")
                            if sn:
                                stats_dict[sn] = s.get("value")
                                stats_str[sn] = s.get("displayValue", s.get("summary", ""))

                        wins = float(stats_dict.get("wins", 0) or 0)
                        losses = float(stats_dict.get("losses", 0) or 0)
                        pj = wins + losses
                        partidos_max = max(partidos_max, pj)
                        pj_safe = max(1.0, pj)

                        pf = float(stats_dict.get("pointsFor", 0) or 0)
                        pa = float(stats_dict.get("pointsAgainst", 0) or 0)
                        ppg = float(stats_dict.get("avgPointsFor") or (pf / pj_safe if pf > 0 else pts_default))
                        oppg = float(stats_dict.get("avgPointsAgainst") or (pa / pj_safe if pa > 0 else pts_default))

                        rec_home = stats_str.get("Home", stats_str.get("homeRecord", "0-0"))
                        rec_away = stats_str.get("Road", stats_str.get("awayRecord", "0-0"))
                        racha = stats_str.get("streak", "-")

                        win_pct_gral = (wins / pj_safe) if pj > 0 else 0.50
                        win_pct_home = parse_win_pct(rec_home, min(0.85, win_pct_gral + 0.08))
                        win_pct_away = parse_win_pct(rec_away, max(0.15, win_pct_gral - 0.08))

                        ajuste_loc = 1.0 + ((win_pct_home - win_pct_gral) * 0.14)
                        ajuste_vis = 1.0 + ((win_pct_away - win_pct_gral) * 0.14)

                        equipos_temp[nombre] = {
                            "id": team_info.get("id", nombre),
                            "pj": int(pj),
                            "ppg": ppg,
                            "oppg": oppg,
                            "ppg_home": ppg * ajuste_loc,
                            "oppg_home": oppg / ajuste_loc,
                            "ppg_away": ppg * ajuste_vis,
                            "oppg_away": oppg / ajuste_vis,
                            "rec_total": f"{int(wins)}-{int(losses)}",
                            "rec_home": rec_home,
                            "rec_away": rec_away,
                            "streak": racha
                        }
                if equipos_temp and partidos_max >= 3:
                    equipos_db = equipos_temp
                    break
                elif equipos_temp and not equipos_db:
                    equipos_db = equipos_temp
        except Exception:
            continue

    url_sb = f"https://site.api.espn.com/apis/site/v2/sports/{ruta}/scoreboard"
    proximos = []
    try:
        r_sb = requests.get(url_sb, timeout=10)
        if r_sb.status_code == 200:
            for ev in r_sb.json().get("events", []):
                comps = ev.get("competitions", [{}])[0].get("competitors", [])
                if len(comps) == 2:
                    home_c = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
                    away_c = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
                    h_name = home_c.get("team", {}).get("displayName")
                    a_name = away_c.get("team", {}).get("displayName")
                    fecha_raw = ev.get("date", "")
                    try:
                        dt = datetime.fromisoformat(fecha_raw.replace("Z", "+00:00"))
                        f_str = dt.strftime("%d/%b %H:%M")
                    except Exception:
                        f_str = "Próximo"

                    for nom_eq in (h_name, a_name):
                        if nom_eq and nom_eq not in equipos_db:
                            equipos_db[nom_eq] = {
                                "id": nom_eq, "pj": 1, "ppg": pts_default, "oppg": pts_default,
                                "ppg_home": pts_default * 1.02, "oppg_home": pts_default * 0.98,
                                "ppg_away": pts_default * 0.98, "oppg_away": pts_default * 1.02,
                                "rec_total": "0-0", "rec_home": "0-0", "rec_away": "0-0", "streak": "-"
                            }
                    if h_name and a_name:
                        proximos.append({
                            "etiqueta": f"📅 {f_str} | {h_name} vs {a_name}",
                            "home": h_name,
                            "away": a_name
                        })
    except Exception:
        pass

    return equipos_db, proximos


@st.cache_data(ttl=1800)
def obtener_cartelera_proximos_partidos(codigo_liga, api_key):
    if codigo_liga == "MLS":
        return []
    headers = {"X-Auth-Token": api_key}
    url = f"https://api.football-data.org/v4/competitions/{codigo_liga}/matches?status=SCHEDULED,TIMED"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        proximos = []
        for m in r.json().get("matches", [])[:25]:
            home = m.get("homeTeam", {}).get("name")
            away = m.get("awayTeam", {}).get("name")
            fecha_utc = m.get("utcDate", "")
            if home and away:
                try:
                    dt = datetime.fromisoformat(fecha_utc.replace("Z", "+00:00"))
                    fecha_str = dt.strftime("%d/%b %H:%M")
                except Exception:
                    fecha_str = "Próximo"
                proximos.append({
                    "etiqueta": f"📅 {fecha_str} | {home} vs {away}",
                    "home": home,
                    "away": away
                })
        return proximos
    except Exception:
        return []


def extraer_float_seguro(fila, cols):
    for c in cols:
        v = fila.get(c, "").strip()
        try:
            if v and float(v) > 1.0:
                return float(v)
        except ValueError:
            continue
    return None


@st.cache_data(ttl=3600)
def cargar_datos_liga_y_csv(codigo_liga, api_key):
    headers = {"X-Auth-Token": api_key}
    base_url = "https://api.football-data.org/v4"
    tablas = {"TOTAL": {}, "HOME": {}, "AWAY": {}}
    stats_db = {}
    partidos_hist = []

    if codigo_liga == "MLS":
        url_mls = "https://www.football-data.co.uk/new/USA.csv"
        resp = requests.get(url_mls, timeout=15)
        texto_csv = resp.content.decode("utf-8-sig", errors="ignore")
        lector = list(csv.DictReader(io.StringIO(texto_csv)))
        validos = [
            f for f in lector
            if f.get("Home") and f.get("Away")
            and f.get("HG", "").strip().isdigit()
            and f.get("AG", "").strip().isdigit()
        ]
        partidos_hist = [
            {
                "home": p["Home"].strip(), "away": p["Away"].strip(),
                "hg": int(p["HG"]), "ag": int(p["AG"]),
                "odds_h": extraer_float_seguro(p, ["AvgH", "PSCH"]),
                "odds_o25": None, "odds_u25": None
            }
            for p in validos[-450:]
        ]
        ult_temp = validos[-1].get("Season")
        temp_act = [f for f in validos if f.get("Season") == ult_temp]
        if len(temp_act) < 30:
            temp_act = validos[-250:]

        conteo = {"HOME": {}, "AWAY": {}, "TOTAL": {}}
        for p in temp_act:
            h, a = p["Home"].strip(), p["Away"].strip()
            hg, ag = int(p["HG"].strip()), int(p["AG"].strip())
            for eq in (h, a):
                for t in ("HOME", "AWAY", "TOTAL"):
                    if eq not in conteo[t]:
                        conteo[t][eq] = {"pj": 0, "gf": 0, "gc": 0}
            conteo["HOME"][h]["pj"] += 1; conteo["HOME"][h]["gf"] += hg; conteo["HOME"][h]["gc"] += ag
            conteo["AWAY"][a]["pj"] += 1; conteo["AWAY"][a]["gf"] += ag; conteo["AWAY"][a]["gc"] += hg
            conteo["TOTAL"][h]["pj"] += 1; conteo["TOTAL"][h]["gf"] += hg; conteo["TOTAL"][h]["gc"] += ag
            conteo["TOTAL"][a]["pj"] += 1; conteo["TOTAL"][a]["gf"] += ag; conteo["TOTAL"][a]["gc"] += hg

        for t in ("HOME", "AWAY", "TOTAL"):
            for eq, d in conteo[t].items():
                pj = max(1, d["pj"])
                tablas[t][eq] = {"id": eq, "shortName": eq, "pj": pj, "gf_prom": d["gf"] / pj, "gc_prom": d["gc"] / pj}
    else:
        resp = requests.get(f"{base_url}/competitions/{codigo_liga}/standings", headers=headers, timeout=12)
        if resp.status_code != 200:
            return None, {}, []
        datos = resp.json()
        for bloque in datos.get("standings", []):
            tipo = bloque.get("type")
            if tipo in tablas:
                for fila in bloque.get("table", []):
                    eq_id = fila["team"]["id"]
                    nombre = fila["team"]["name"]
                    corto = fila["team"].get("shortName", nombre)
                    pj = max(1, fila["playedGames"])
                    tablas[tipo][nombre] = {
                        "id": eq_id, "shortName": corto, "pj": pj,
                        "gf_prom": fila["goalsFor"] / pj, "gc_prom": fila["goalsAgainst"] / pj
                    }
        if not tablas["HOME"]:
            tablas["HOME"] = tablas["TOTAL"]
        if not tablas["AWAY"]:
            tablas["AWAY"] = tablas["TOTAL"]

        codigos_csv = list(CSV_LIGAS_EUROPEAS.values()) if codigo_liga == "CL" else [CSV_LIGAS_EUROPEAS.get(codigo_liga, "SP1")]
        acumulado = {}
        for cod_csv in codigos_csv:
            for temp in ["2627", "2526", "2425"]:
                url_csv = f"https://www.football-data.co.uk/mmz4281/{temp}/{cod_csv}.csv"
                try:
                    r = requests.get(url_csv, timeout=8)
                    if r.status_code == 200 and len(r.text) > 500:
                        lector = csv.DictReader(io.StringIO(r.text))
                        leidos = 0
                        for fila in lector:
                            h, a = fila.get("HomeTeam", "").strip(), fila.get("AwayTeam", "").strip()
                            fthg, ftag = fila.get("FTHG", "").strip(), fila.get("FTAG", "").strip()
                            hc, ac = fila.get("HC", ""), fila.get("AC", "")
                            hy, ay = fila.get("HY", ""), fila.get("AY", "")
                            hr, ar = fila.get("HR", "0") or "0", fila.get("AR", "0") or "0"
                            hst, ast = fila.get("HST", "0") or "0", fila.get("AST", "0") or "0"

                            if h and a and fthg.isdigit() and ftag.isdigit():
                                partidos_hist.append({
                                    "home": h, "away": a, "hg": int(fthg), "ag": int(ftag),
                                    "odds_h": extraer_float_seguro(fila, ["B365H", "AvgH"]),
                                    "odds_o25": extraer_float_seguro(fila, ["B365>2.5", "Avg>2.5"]),
                                    "odds_u25": extraer_float_seguro(fila, ["B365<2.5", "Avg<2.5"])
                                })
                            if h and a and hc.isdigit() and ac.isdigit():
                                leidos += 1
                                for eq in (h, a):
                                    if eq not in acumulado:
                                        acumulado[eq] = {"cf": [], "cc": [], "tarj": [], "tiros": []}
                                tarj_tot = float(hy or 0) + float(ay or 0) + float(hr) + float(ar)
                                acumulado[h]["cf"].append(float(hc)); acumulado[h]["cc"].append(float(ac))
                                acumulado[h]["tarj"].append(tarj_tot); acumulado[h]["tiros"].append(float(hst))
                                acumulado[a]["cf"].append(float(ac)); acumulado[a]["cc"].append(float(hc))
                                acumulado[a]["tarj"].append(tarj_tot); acumulado[a]["tiros"].append(float(ast))
                        if leidos >= 10:
                            break
                except Exception:
                    continue

        for eq, d in acumulado.items():
            if d["cf"]:
                stats_db[eq] = {
                    "corners_favor": float(np.mean(d["cf"])),
                    "corners_contra": float(np.mean(d["cc"])),
                    "tarjetas_partido": float(np.mean(d["tarj"])),
                    "tiros_puerta": float(np.mean(d["tiros"]))
                }

    return tablas, stats_db, partidos_hist[-500:]


@st.cache_data(ttl=1800)
def obtener_forma_detallada_y_h2h(codigo_liga, id_eq, id_riv, partidos_mls, api_key):
    alpha = 0.15
    gf_r, gc_r, pesos, racha = [], [], [], []
    gf_home, gc_home, racha_home = [], [], []
    gf_away, gc_away, racha_away = [], [], []
    btts_count, over25_count = 0, 0
    gf_h, gc_h = [], []
    lista_partidos_raw = []

    if codigo_liga == "MLS":
        for p in reversed(partidos_mls):
            if p["home"] == id_eq or p["away"] == id_eq:
                es_loc = (p["home"] == id_eq)
                es_riv = (p["home"] == id_riv or p["away"] == id_riv)
                gf = p["hg"] if es_loc else p["ag"]
                gc = p["ag"] if es_loc else p["hg"]
                lista_partidos_raw.append((es_loc, gf, gc, es_riv))
    else:
        headers = {"X-Auth-Token": api_key}
        url = f"https://api.football-data.org/v4/teams/{id_eq}/matches?status=FINISHED&limit=35"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                partidos = sorted(r.json().get("matches", []), key=lambda x: x.get("utcDate", ""), reverse=True)
                for m in partidos:
                    sc = m.get("score", {}).get("fullTime", {})
                    gh, ga = sc.get("home"), sc.get("away")
                    if gh is None or ga is None:
                        continue
                    hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
                    es_loc = (hid == id_eq)
                    es_riv = (hid == id_riv or aid == id_riv)
                    gf, gc = (gh if es_loc else ga), (ga if es_loc else gh)
                    lista_partidos_raw.append((es_loc, gf, gc, es_riv))
        except Exception:
            pass

    for es_loc, gf, gc, es_riv in lista_partidos_raw:
        etiqueta_res = f"{'G' if gf>gc else ('E' if gf==gc else 'P')}({gf}-{gc})"
        if len(gf_r) < 10:
            pesos.append(np.exp(-alpha * len(gf_r)))
            gf_r.append(gf); gc_r.append(gc)
            if gf >= 1 and gc >= 1:
                btts_count += 1
            if (gf + gc) >= 3:
                over25_count += 1
            if len(racha) < 5:
                racha.append(etiqueta_res)

        if es_loc and len(gf_home) < 6:
            gf_home.append(gf); gc_home.append(gc)
            if len(racha_home) < 5:
                racha_home.append(etiqueta_res)
        elif (not es_loc) and len(gf_away) < 6:
            gf_away.append(gf); gc_away.append(gc)
            if len(racha_away) < 5:
                racha_away.append(etiqueta_res)

        if es_riv:
            gf_h.append(gf); gc_h.append(gc)

    if not gf_r:
        return None

    gf_ewma_gral = float(np.average(gf_r, weights=pesos))
    gc_ewma_gral = float(np.average(gc_r, weights=pesos))
    pesos_h = [np.exp(-alpha * i) for i in range(len(gf_home))]
    pesos_a = [np.exp(-alpha * i) for i in range(len(gf_away))]

    return {
        "gf_ewma": gf_ewma_gral,
        "gc_ewma": gc_ewma_gral,
        "gf_ewma_home": float(np.average(gf_home, weights=pesos_h)) if gf_home else gf_ewma_gral,
        "gc_ewma_home": float(np.average(gc_home, weights=pesos_h)) if gc_home else gc_ewma_gral,
        "gf_ewma_away": float(np.average(gf_away, weights=pesos_a)) if gf_away else gf_ewma_gral,
        "gc_ewma_away": float(np.average(gc_away, weights=pesos_a)) if gc_away else gc_ewma_gral,
        "racha_5": " ".join(racha),
        "racha_home_5": " ".join(racha_home) if racha_home else "Sin datos",
        "racha_away_5": " ".join(racha_away) if racha_away else "Sin datos",
        "n_analizados": len(gf_r),
        "btts_recientes": btts_count,
        "over25_recientes": over25_count,
        "h2h_n": len(gf_h),
        "gf_h2h": float(np.mean(gf_h)) if gf_h else None,
        "gc_h2h": float(np.mean(gc_h)) if gc_h else None
    }


def generar_matriz_dc(l_loc, l_vis, rho=-0.05, max_g=10):
    p_loc = [poisson.pmf(i, l_loc) for i in range(max_g)]
    p_vis = [poisson.pmf(j, l_vis) for j in range(max_g)]
    mat = np.outer(p_loc, p_vis)
    mat[0, 0] *= max(0.1, 1.0 - (l_loc * l_vis * rho))
    mat[0, 1] *= max(0.1, 1.0 + (l_loc * rho))
    mat[1, 0] *= max(0.1, 1.0 + (l_vis * rho))
    mat[1, 1] *= max(0.1, 1.0 - rho)
    return mat / np.sum(mat)


def entrenar_liga_ml(partidos_hist):
    if len(partidos_hist) < 30:
        return 0.60, 1.00, -0.05, None
    hist_eq, ejemplos = {}, []
    m_loc_g = float(np.mean([p["hg"] for p in partidos_hist]))
    m_vis_g = float(np.mean([p["ag"] for p in partidos_hist]))

    for p in partidos_hist:
        h, a, hg, ag = p["home"], p["away"], p["hg"], p["ag"]
        if h in hist_eq and a in hist_eq and len(hist_eq[h]["gf"]) >= 4 and len(hist_eq[a]["gf"]) >= 4:
            ejemplos.append({
                "hg": hg, "ag": ag,
                "h_gf_t": np.mean(hist_eq[h]["gf"]), "h_gc_t": np.mean(hist_eq[h]["gc"]),
                "a_gf_t": np.mean(hist_eq[a]["gf"]), "a_gc_t": np.mean(hist_eq[a]["gc"]),
                "h_gf_u": hist_eq[h]["gf"][-6:], "h_gc_u": hist_eq[h]["gc"][-6:],
                "a_gf_u": hist_eq[a]["gf"][-6:], "a_gc_u": hist_eq[a]["gc"][-6:],
                "odds_o25": p.get("odds_o25"), "odds_u25": p.get("odds_u25"), "odds_h": p.get("odds_h")
            })
        for eq, gfv, gcv in [(h, hg, ag), (a, ag, hg)]:
            if eq not in hist_eq:
                hist_eq[eq] = {"gf": [], "gc": []}
            hist_eq[eq]["gf"].append(gfv); hist_eq[eq]["gc"].append(gcv)

    if len(ejemplos) < 20:
        return 0.60, 1.00, -0.05, None

    corte = int(len(ejemplos) * 0.70)
    train_s, test_s = ejemplos[:corte], ejemplos[corte:]

    def eval_p(ex, wt, floc):
        wf = 1.0 - wt
        llb = (ex["h_gf_t"] / m_loc_g) * (ex["a_gc_t"] / m_loc_g) * m_loc_g
        lvb = (ex["a_gf_t"] / m_vis_g) * (ex["h_gc_t"] / m_vis_g) * m_vis_g
        llr = (np.mean(ex["h_gf_u"]) + np.mean(ex["a_gc_u"])) / 2.0
        lvr = (np.mean(ex["a_gf_u"]) + np.mean(ex["h_gc_u"])) / 2.0
        lh = (0.85 * max(0.35, ((wt * llb) + (wf * llr)) * floc)) + (0.15 * m_loc_g)
        la = (0.85 * max(0.35, (wt * lvb) + (wf * lvr))) + (0.15 * m_vis_g)
        return lh, la

    def loss_f(params):
        wt, floc, rho = params
        loss = 0.0
        for ex in train_s:
            lh, la = eval_p(ex, wt, floc)
            hg, ag = ex["hg"], ex["ag"]
            pb = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
            tau = (1.0 - lh*la*rho) if (hg==0 and ag==0) else ((1.0 + lh*rho) if (hg==0 and ag==1) else ((1.0 + la*rho) if (hg==1 and ag==0) else ((1.0 - rho) if (hg==1 and ag==1) else 1.0)))
            loss -= np.log(max(1e-6, pb * max(0.1, tau)))
        return loss / len(train_s)

    res = minimize(loss_f, x0=[0.60, 1.00, -0.04], bounds=[(0.35, 0.85), (0.88, 1.15), (-0.15, 0.10)], method="L-BFGS-B")
    wt_opt, floc_opt, rho_opt = (float(res.x[0]), float(res.x[1]), float(res.x[2])) if res.success else (0.60, 1.00, -0.05)

    ap_n, ap_w, unid = 0, 0, 0.0
    for ex in test_s:
        lh, la = eval_p(ex, wt_opt, floc_opt)
        mat = generar_matriz_dc(lh, la, rho_opt)
        po25 = float(sum(mat[i, j] for i in range(10) for j in range(10) if (i + j) >= 3))
        ph = float(np.sum(np.tril(mat, -1)))
        for pm, c_real, gano in [(po25, ex.get("odds_o25"), (ex["hg"]+ex["ag"])>=3), (1-po25, ex.get("odds_u25"), (ex["hg"]+ex["ag"])<=2), (ph, ex.get("odds_h"), ex["hg"]>ex["ag"])]:
            if c_real and 1.45 <= c_real <= 3.20:
                ev_b = ((pm * c_real) - 1.0) * 100
                if 5.0 <= ev_b <= 25.0:
                    ap_n += 1
                    if gano:
                        ap_w += 1; unid += (c_real - 1.0)
                    else:
                        unid -= 1.0

    bt_info = {"apuestas": ap_n, "acierto": (ap_w / ap_n * 100) if ap_n else 0, "roi": (unid / ap_n * 100) if ap_n else 0} if ap_n > 0 else None
    return wt_opt, floc_opt, rho_opt, bt_info


def buscar_det_equipo(nom, corto, lam, stats_db):
    nombres_db = list(stats_db.keys())
    for cand in [corto, nom]:
        for ndb in nombres_db:
            if ndb.lower() in cand.lower() or cand.lower() in ndb.lower():
                return stats_db[ndb], True
        coinc = get_close_matches(cand, nombres_db, n=1, cutoff=0.55)
        if coinc:
            return stats_db[coinc[0]], True
    return {"corners_favor": 3.7 + lam * 1.15, "corners_contra": 4.7, "tarjetas_partido": 4.2, "tiros_puerta": 2.9 + lam * 1.4}, False


def ejecutar_interfaz_nba_nfl(codigo_liga, nombre_liga, bankroll_usuario, castigo_loc, castigo_vis):
    with st.spinner(f"Conectando con ESPN API en vivo para {nombre_liga}..."):
        equipos_db, proximos_partidos = cargar_datos_nba_nfl_espn(codigo_liga)

    lista_equipos = sorted(list(equipos_db.keys()))
    if not lista_equipos:
        st.error("No se encontraron equipos activos en este momento.")
        return

    st.subheader(f"📅 Cartelera de Partidos Disponibles — {nombre_liga}")
    col_sel1, col_sel2 = st.columns([2, 2])

    with col_sel1:
        if proximos_partidos:
            opciones_cartelera = ["🔍 Seleccionar equipos manualmente..."] + [p["etiqueta"] for p in proximos_partidos]
            partido_elegido = st.selectbox("Elige un partido programado de la jornada (1 clic):", opciones_cartelera)
        else:
            partido_elegido = "🔍 Seleccionar equipos manualmente..."
            st.info("Selecciona los equipos en los menús de la derecha.")

    eq_local_def = lista_equipos[0]
    eq_visita_def = lista_equipos[min(1, len(lista_equipos) - 1)]

    if partido_elegido != "🔍 Seleccionar equipos manualmente...":
        datos_p = next(p for p in proximos_partidos if p["etiqueta"] == partido_elegido)
        if datos_p["home"] in lista_equipos:
            eq_local_def = datos_p["home"]
        if datos_p["away"] in lista_equipos:
            eq_visita_def = datos_p["away"]

    with col_sel2:
        c_eq1, c_eq2 = st.columns(2)
        eq_local = c_eq1.selectbox("🏠 Equipo LOCAL:", lista_equipos, index=lista_equipos.index(eq_local_def))
        eq_visita = c_eq2.selectbox("✈️ Equipo VISITANTE:", lista_equipos, index=lista_equipos.index(eq_visita_def))

    st_loc = equipos_db[eq_local]
    st_vis = equipos_db[eq_visita]

    hca = 2.8 if codigo_liga == "NBA" else 2.3
    sigma_margen = 11.5 if codigo_liga == "NBA" else 13.2
    sigma_total = 15.5 if codigo_liga == "NBA" else 13.8

    pts_loc_base = (0.60 * ((st_loc["ppg_home"] + st_vis["oppg_away"]) / 2.0)) + (0.40 * ((st_loc["ppg"] + st_vis["oppg"]) / 2.0)) + (hca / 2.0)
    pts_vis_base = (0.60 * ((st_vis["ppg_away"] + st_loc["oppg_home"]) / 2.0)) + (0.40 * ((st_vis["ppg"] + st_loc["oppg"]) / 2.0)) - (hca / 2.0)

    pts_loc = pts_loc_base * (1.0 - castigo_loc / 100.0)
    pts_vis = pts_vis_base * (1.0 - castigo_vis / 100.0)
    pts_tot = pts_loc + pts_vis
    margen_esperado = pts_loc - pts_vis

    prob_gana_loc = float(1.0 - norm.cdf(0.0, loc=margen_esperado, scale=sigma_margen))
    prob_gana_vis = 1.0 - prob_gana_loc

    nom_loc_c = nombre_corto_us(eq_local)
    nom_vis_c = nombre_corto_us(eq_visita)
    ganador_c = nom_loc_c if prob_gana_loc >= 0.5 else nom_vis_c
    prob_ganador = max(prob_gana_loc, prob_gana_vis)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    mostrar_tarjeta_quant(m1, "🔢 Puntos Esperados Totales", f"{pts_tot:.1f} Pts", f"{nom_loc_c}: {pts_loc:.1f} | {nom_vis_c}: {pts_vis:.1f}")
    mostrar_tarjeta_quant(m2, "🏆 Favorito del Modelo", f"{ganador_c} ({prob_ganador*100:.1f}%)", f"{nom_loc_c}: {prob_gana_loc*100:.1f}% | {nom_vis_c}: {prob_gana_vis*100:.1f}%")
    mostrar_tarjeta_quant(m3, "📏 Hándicap (Spread) Justo", f"{nom_loc_c} {-margen_esperado:+.1f}", f"Marcador Est.: {int(round(pts_loc))} - {int(round(pts_vis))}")
    mostrar_tarjeta_quant(m4, "🏠 Forma Local vs ✈️ Visita", f"Casa: {st_loc['rec_home']} | Ruta: {st_vis['rec_away']}", f"General: {st_loc['rec_total']} vs {st_vis['rec_total']}")

    st.caption(
        f"🔥 **Promedios {eq_local} (Local):** `{st_loc['ppg_home']:.1f} Pts Anotados / {st_loc['oppg_home']:.1f} Recibidos` (Racha: {st_loc['streak']})  |  "
        f"🔥 **Promedios {eq_visita} (Visita):** `{st_vis['ppg_away']:.1f} Pts Anotados / {st_vis['oppg_away']:.1f} Recibidos` (Racha: {st_vis['streak']})"
    )

    linea_tot_central = round(pts_tot * 2) / 2.0
    paso_pts = 4.0 if codigo_liga == "NBA" else 3.0
    spread_central = round(abs(margen_esperado)) + 0.5

    mercados_deporte_us = [
        (f"Ganador (Moneyline): {eq_local}", prob_gana_loc),
        (f"Ganador (Moneyline): {eq_visita}", prob_gana_vis),
        (f"Hándicap: {eq_local} -{spread_central:.1f} Pts", float(1.0 - norm.cdf(spread_central, loc=margen_esperado, scale=sigma_margen))),
        (f"Hándicap: {eq_visita} +{spread_central:.1f} Pts", float(norm.cdf(spread_central, loc=margen_esperado, scale=sigma_margen))),
        (f"Hándicap Alternativo: {eq_local} +3.5 Pts", float(1.0 - norm.cdf(-3.5, loc=margen_esperado, scale=sigma_margen))),
        (f"Hándicap Alternativo: {eq_visita} +7.5 Pts", float(norm.cdf(7.5, loc=margen_esperado, scale=sigma_margen))),
        (f"Puntos Totales: Más de {linea_tot_central - paso_pts:.1f} (Over)", float(1.0 - norm.cdf(linea_tot_central - paso_pts, loc=pts_tot, scale=sigma_total))),
        (f"Puntos Totales: Más de {linea_tot_central:.1f} (Over)", float(1.0 - norm.cdf(linea_tot_central, loc=pts_tot, scale=sigma_total))),
        (f"Puntos Totales: Menos de {linea_tot_central:.1f} (Under)", float(norm.cdf(linea_tot_central, loc=pts_tot, scale=sigma_total))),
        (f"Puntos Totales: Menos de {linea_tot_central + paso_pts:.1f} (Under)", float(norm.cdf(linea_tot_central + paso_pts, loc=pts_tot, scale=sigma_total))),
        (f"{eq_local}: Más de {round(pts_loc)-0.5:.1f} Puntos", float(1.0 - norm.cdf(round(pts_loc)-0.5, loc=pts_loc, scale=sigma_total*0.65))),
        (f"{eq_visita}: Más de {round(pts_vis)-0.5:.1f} Puntos", float(1.0 - norm.cdf(round(pts_vis)-0.5, loc=pts_vis, scale=sigma_total*0.65))),
    ]

    filas_us, opciones_us = [], {}
    for nom_m, p_m in mercados_deporte_us:
        p_m = max(0.01, min(0.99, p_m))
        filas_us.append({
            "Mercado": nom_m,
            "Probabilidad Real (%)": round(p_m * 100, 1),
            "Cuota Justa": round(1.0 / p_m, 2),
            "Cuota Mínima (+5% EV)": round(1.05 / p_m, 2)
        })
        opciones_us[nom_m] = p_m

    col_t, col_p = st.columns([1.35, 1])
    with col_t:
        st.subheader(f"📊 Mercados Calculados — {eq_local} vs {eq_visita}")
        st.dataframe(pd.DataFrame(filas_us), use_container_width=True, hide_index=True, height=460)

    with col_p:
        st.subheader("🎯 Evaluador de Parlay / SGP (Draftea)")
        sel_us = st.multiselect("Selecciona los mercados a combinar:", list(opciones_us.keys()), default=[mercados_deporte_us[0][0], mercados_deporte_us[6][0]])
        cuota_d = st.number_input("💵 Momio en Draftea:", min_value=1.01, value=2.10, step=0.05)
        if sel_us:
            prob_tot = 1.0
            for idx_s, s in enumerate(sel_us):
                pen = 0.95 if idx_s > 0 else 1.0
                prob_tot *= (opciones_us[s] * pen)
            c_justa = 1.0 / prob_tot
            c_min = 1.05 / prob_tot
            ev = ((prob_tot * cuota_d) - 1.0) * 100
            b = cuota_d - 1.0
            kelly_p = ((b * prob_tot) - (1.0 - prob_tot)) / b if b > 0 else 0.0
            stake_pct = min(0.015, max(0.0, kelly_p * 0.125))
            k1, k2, k3 = st.columns(3)
            k1.metric("Prob. Conjunta", f"{prob_tot*100:.1f}%")
            k2.metric("Cuota Justa", f"{c_justa:.2f}")
            k3.metric("Cuota Mín (+5% EV)", f"{c_min:.2f}")
            if ev >= 5.0:
                st.success(f"✅ **APOSTAR (+EV: {ev:+.2f}%)** — Sugerido (1/8 Kelly): `${bankroll_usuario*stake_pct:.2f} MXN` ({stake_pct*100:.2f}%)")
            elif ev > 0:
                st.warning(f"⚠️ **ZONA GRIS (EV: {ev:+.2f}%)** — Exige cuota >= `{c_min:.2f}`")
            else:
                st.error(f"❌ **NO APOSTAR (EV: {ev:+.2f}%)** — Exige cuota >= `{c_min:.2f}`")


def ejecutar_interfaz_futbol(codigo_liga, nombre_liga, bankroll_usuario, castigo_loc, castigo_vis):
    with st.spinner(f"Sincronizando base de datos y entrenando modelo para {nombre_liga}..."):
        tablas, stats_db, partidos_hist = cargar_datos_liga_y_csv(codigo_liga, API_KEY)
        if tablas is None:
            st.error("Error conectando con football-data.org. Verifica que hayas puesto tu API_KEY en la línea 13.")
            return
        proximos_partidos = obtener_cartelera_proximos_partidos(codigo_liga, API_KEY)
        w_temp, f_loc_ml, rho_dc, bt_res = entrenar_liga_ml(partidos_hist)

    lista_equipos = sorted(list(tablas["HOME"].keys()))

    st.subheader(f"📅 Cartelera de Partidos Disponibles — {nombre_liga}")
    col_sel1, col_sel2 = st.columns([2, 2])

    with col_sel1:
        if proximos_partidos:
            opciones_cartelera = ["🔍 Seleccionar equipos manualmente..."] + [p["etiqueta"] for p in proximos_partidos]
            partido_elegido = st.selectbox("Elige un partido programado de la jornada (1 clic):", opciones_cartelera)
        else:
            partido_elegido = "🔍 Seleccionar equipos manualmente..."
            st.info("Elige los equipos directamente en las listas de la derecha.")

    eq_local_def = lista_equipos[0]
    eq_visita_def = lista_equipos[min(1, len(lista_equipos) - 1)]

    if partido_elegido != "🔍 Seleccionar equipos manualmente...":
        datos_p = next(p for p in proximos_partidos if p["etiqueta"] == partido_elegido)
        if datos_p["home"] in lista_equipos:
            eq_local_def = datos_p["home"]
        if datos_p["away"] in lista_equipos:
            eq_visita_def = datos_p["away"]

    with col_sel2:
        c_eq1, c_eq2 = st.columns(2)
        eq_local = c_eq1.selectbox("🏠 Equipo LOCAL:", lista_equipos, index=lista_equipos.index(eq_local_def))
        eq_visita = c_eq2.selectbox("✈️ Equipo VISITANTE:", lista_equipos, index=lista_equipos.index(eq_visita_def))

    st_loc = tablas["HOME"][eq_local]
    st_vis = tablas["AWAY"].get(eq_visita, tablas["TOTAL"][eq_visita])

    forma_loc = obtener_forma_detallada_y_h2h(codigo_liga, st_loc["id"], st_vis["id"], partidos_hist, API_KEY)
    forma_vis = obtener_forma_detallada_y_h2h(codigo_liga, st_vis["id"], st_loc["id"], partidos_hist, API_KEY)

    m_loc = float(np.mean([x["gf_prom"] for x in tablas["HOME"].values()]))
    m_vis = float(np.mean([x["gf_prom"] for x in tablas["AWAY"].values()]))

    l_loc_liga = (st_loc["gf_prom"] / m_loc) * (st_vis["gc_prom"] / m_loc) * m_loc
    l_vis_liga = (st_vis["gf_prom"] / m_vis) * (st_loc["gc_prom"] / m_vis) * m_vis

    if forma_loc and forma_vis:
        gf_loc_esp = (0.65 * forma_loc["gf_ewma_home"]) + (0.35 * forma_loc["gf_ewma"])
        gc_loc_esp = (0.65 * forma_loc["gc_ewma_home"]) + (0.35 * forma_loc["gc_ewma"])
        gf_vis_esp = (0.65 * forma_vis["gf_ewma_away"]) + (0.35 * forma_vis["gf_ewma"])
        gc_vis_esp = (0.65 * forma_vis["gc_ewma_away"]) + (0.35 * forma_vis["gc_ewma"])

        l_loc_f = (gf_loc_esp + gc_vis_esp) / 2.0
        l_vis_f = (gf_vis_esp + gc_loc_esp) / 2.0
    else:
        l_loc_f, l_vis_f = l_loc_liga, l_vis_liga

    w_forma = 1.0 - w_temp
    lam_loc = ((w_temp * l_loc_liga) + (w_forma * l_loc_f)) * f_loc_ml
    lam_vis = (w_temp * l_vis_liga) + (w_forma * l_vis_f)

    if forma_loc and forma_loc["gf_h2h"] is not None:
        lam_loc = (0.88 * lam_loc) + (0.12 * forma_loc["gf_h2h"])
        lam_vis = (0.88 * lam_vis) + (0.12 * forma_loc["gc_h2h"])

    lam_loc = max(0.35, ((0.85 * lam_loc) + (0.15 * m_loc)) * (1.0 - castigo_loc / 100.0))
    lam_vis = max(0.35, ((0.85 * lam_vis) + (0.15 * m_vis)) * (1.0 - castigo_vis / 100.0))

    matriz = generar_matriz_dc(lam_loc, lam_vis, rho=rho_dc)

    p_gana_loc = float(np.sum(np.tril(matriz, -1)))
    p_empate = float(np.sum(np.diag(matriz)))
    p_gana_vis = float(np.sum(np.triu(matriz, 1)))
    p_btts_si = float(sum(matriz[i, j] for i in range(1, 10) for j in range(1, 10)))
    p_over25 = float(sum(matriz[i, j] for i in range(10) for j in range(10) if (i + j) >= 3))

    marcadores_ordenados = sorted(
        [(i, j, float(matriz[i, j])) for i in range(7) for j in range(7)],
        key=lambda x: x[2],
        reverse=True
    )
    top1_h, top1_a, top1_p = marcadores_ordenados[0]
    top2_h, top2_a, top2_p = marcadores_ordenados[1]
    top3_h, top3_a, top3_p = marcadores_ordenados[2]

    nom_loc_corto = st_loc["shortName"]
    nom_vis_corto = st_vis["shortName"]
    opciones_1x2 = [(nom_loc_corto, p_gana_loc), ("Empate", p_empate), (nom_vis_corto, p_gana_vis)]
    favorito_nombre, favorito_prob = max(opciones_1x2, key=lambda x: x[1])

    det_loc, real_loc = buscar_det_equipo(eq_local, st_loc["shortName"], lam_loc, stats_db)
    det_vis, real_vis = buscar_det_equipo(eq_visita, st_vis["shortName"], lam_vis, stats_db)

    l_c_loc = (det_loc["corners_favor"] + det_vis["corners_contra"]) / 2.0
    l_c_vis = (det_vis["corners_favor"] + det_loc["corners_contra"]) / 2.0
    l_c_tot = l_c_loc + l_c_vis
    l_tarj = (det_loc["tarjetas_partido"] + det_vis["tarjetas_partido"]) / 2.0

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    mostrar_tarjeta_quant(m1, "⚽ Goles Esperados (xG)", f"{lam_loc + lam_vis:.2f} Totales", f"{nom_loc_corto}: {lam_loc:.2f} | {nom_vis_corto}: {lam_vis:.2f}")
    mostrar_tarjeta_quant(m2, "🚩 Córners Esperados", f"{l_c_tot:.1f} Totales", f"{nom_loc_corto}: {l_c_loc:.1f} | {nom_vis_corto}: {l_c_vis:.1f}")
    mostrar_tarjeta_quant(m3, "🟨 Tarjetas Esperadas", f"{l_tarj:.1f} Tarjetas", f"Tiros a puerta: {det_loc['tiros_puerta']:.1f} vs {det_vis['tiros_puerta']:.1f}")
    if bt_res:
        mostrar_tarjeta_quant(m4, "📈 Backtest Real en esta Liga", f"ROI: {bt_res['roi']:+.1f}%", f"Acierto: {bt_res['acierto']:.1f}% ({bt_res['apuestas']} ap.)")
    else:
        mostrar_tarjeta_quant(m4, "🧠 Calibración ML", f"Temp: {w_temp*100:.0f}% / Forma: {w_forma*100:.0f}%", f"Dixon-Coles Rho: {rho_dc:+.3f}")

    p1, p2, p3, p4 = st.columns(4)
    mostrar_tarjeta_quant(
        p1, "🏆 Pronóstico Ganador (1X2)",
        f"{favorito_nombre} ({favorito_prob*100:.1f}%)",
        f"1: {p_gana_loc*100:.0f}% | X: {p_empate*100:.0f}% | 2: {p_gana_vis*100:.0f}%"
    )
    mostrar_tarjeta_quant(
        p2, "🎯 Marcador Más Probable",
        f"{top1_h} - {top1_a} ({top1_p*100:.1f}%)",
        f"2º: {top2_h}-{top2_a} ({top2_p*100:.1f}%) | 3º: {top3_h}-{top3_a} ({top3_p*100:.1f}%)"
    )
    btts_label = f"SÍ ({p_btts_si*100:.1f}%)" if p_btts_si >= 0.50 else f"NO ({(1-p_btts_si)*100:.1f}%)"
    btts_hist_str = (
        f"BTTS rec.: {nom_loc_corto} {forma_loc['btts_recientes']}/{forma_loc['n_analizados']} | {nom_vis_corto} {forma_vis['btts_recientes']}/{forma_vis['n_analizados']}"
        if (forma_loc and forma_vis) else f"Prob. Ambos Anotan: {p_btts_si*100:.1f}%"
    )
    mostrar_tarjeta_quant(p3, "🤝 Ambos Anotan (BTTS)", btts_label, btts_hist_str)

    o25_label = f"Más 2.5 ({p_over25*100:.1f}%)" if p_over25 >= 0.50 else f"Menos 2.5 ({(1-p_over25)*100:.1f}%)"
    o25_hist_str = (
        f"+2.5 rec.: {nom_loc_corto} {forma_loc['over25_recientes']}/{forma_loc['n_analizados']} | {nom_vis_corto} {forma_vis['over25_recientes']}/{forma_vis['n_analizados']}"
        if (forma_loc and forma_vis) else f"Prob. +2.5 Goles: {p_over25*100:.1f}%"
    )
    mostrar_tarjeta_quant(p4, "🔥 Tendencia de Goles (2.5)", o25_label, o25_hist_str)

    if forma_loc and forma_vis:
        st.caption(
            f"🏠 **{eq_local} jugando en CASA:** `{forma_loc['racha_home_5']}` (EWMA Casa: **{forma_loc['gf_ewma_home']:.2f} GF / {forma_loc['gc_ewma_home']:.2f} GC**) — General: `{forma_loc['racha_5']}`  \n"
            f"✈️ **{eq_visita} jugando de VISITA:** `{forma_vis['racha_away_5']}` (EWMA Visita: **{forma_vis['gf_ewma_away']:.2f} GF / {forma_vis['gc_ewma_away']:.2f} GC**) — General: `{forma_vis['racha_5']}`  |  "
            f"🔍 **Datos Córners:** {'✅ Reales CSV' if (real_loc and real_vis) else '⚠️ Estimados'}"
        )

    mercados_extra = [
        ("Córners Totales: Más de 8.5", 1.0 - poisson.cdf(8, l_c_tot)),
        ("Córners Totales: Más de 9.5", 1.0 - poisson.cdf(9, l_c_tot)),
        ("Córners Totales: Más de 10.5", 1.0 - poisson.cdf(10, l_c_tot)),
        ("Córners Totales: Menos de 10.5", poisson.cdf(10, l_c_tot)),
        (f"{eq_local}: Más de 3.5 Córners", 1.0 - poisson.cdf(3, l_c_loc)),
        (f"{eq_local}: Más de 4.5 Córners", 1.0 - poisson.cdf(4, l_c_loc)),
        (f"{eq_visita}: Más de 3.5 Córners", 1.0 - poisson.cdf(3, l_c_vis)),
        (f"{eq_visita}: Más de 4.5 Córners", 1.0 - poisson.cdf(4, l_c_vis)),
        ("Tarjetas Totales: Más de 3.5", 1.0 - poisson.cdf(3, l_tarj)),
        ("Tarjetas Totales: Más de 4.5", 1.0 - poisson.cdf(4, l_tarj)),
        ("Tarjetas Totales: Más de 5.5", 1.0 - poisson.cdf(5, l_tarj)),
        ("Tarjetas Totales: Menos de 4.5", poisson.cdf(4, l_tarj)),
    ]

    def prob_comb_goles(indices):
        prob = 0.0
        for i in range(10):
            for j in range(10):
                if all(CATALOGO_MERCADOS_GOLES[idx][1](i, j) for idx in indices):
                    prob += matriz[i][j]
        return float(prob)

    filas_tabla, opciones_parlay = [], {}
    for idx, (nom_m, _) in enumerate(CATALOGO_MERCADOS_GOLES):
        p = prob_comb_goles([idx])
        filas_tabla.append({
            "Categoría": "⚽ Goles / Resultado",
            "Mercado": nom_m,
            "Probabilidad Real (%)": round(p * 100, 1),
            "Cuota Justa": round(1.0 / p, 2) if p > 0 else 99.0,
            "Cuota Mínima (+5% EV)": round(1.05 / p, 2) if p > 0 else 99.0
        })
        opciones_parlay[nom_m] = ("GOLES", idx, p)

    for idx_e, (nom_e, p_e) in enumerate(mercados_extra):
        filas_tabla.append({
            "Categoría": "🚩 Córners / 🟨 Tarjetas",
            "Mercado": nom_e,
            "Probabilidad Real (%)": round(p_e * 100, 1),
            "Cuota Justa": round(1.0 / p_e, 2) if p_e > 0 else 99.0,
            "Cuota Mínima (+5% EV)": round(1.05 / p_e, 2) if p_e > 0 else 99.0
        })
        opciones_parlay[nom_e] = ("EXTRA", idx_e, p_e)

    col_t, col_p = st.columns([1.35, 1])
    with col_t:
        st.subheader("📊 Tabla de Probabilidades y Cuotas Mínimas")
        st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True, height=520)

    with col_p:
        st.subheader("🎯 Creador de Parlays / SGP (Draftea)")
        selecciones = st.multiselect(
            "Haz clic en las patas que quieres combinar en tu boleto:",
            options=list(opciones_parlay.keys()),
            default=["Doble Oportunidad Empate/Visita (X2)", "Más de 2.5 Goles (Over 2.5)", "Ambos Equipos Anotan: Sí (BTTS)"]
        )
        cuota_draftea = st.number_input("💵 ¿Qué momio decimal te paga Draftea?", min_value=1.01, value=2.03, step=0.05)

        if selecciones:
            idx_goles = [opciones_parlay[s][1] for s in selecciones if opciones_parlay[s][0] == "GOLES"]
            idx_extra = [opciones_parlay[s][1] for s in selecciones if opciones_parlay[s][0] == "EXTRA"]

            prob_total = prob_comb_goles(idx_goles) if idx_goles else 1.0
            for ie in idx_extra:
                factor_sgp = 0.94 if (idx_goles or len(idx_extra) > 1) else 1.0
                prob_total *= (mercados_extra[ie][1] * factor_sgp)

            if prob_total <= 0.0001:
                st.error("❌ Esas selecciones se contradicen entre sí (Probabilidad = 0%).")
            else:
                c_justa = 1.0 / prob_total
                c_min_segura = 1.05 / prob_total
                ev = ((prob_total * cuota_draftea) - 1.0) * 100
                b = cuota_draftea - 1.0
                kelly_puro = ((b * prob_total) - (1.0 - prob_total)) / b if b > 0 else 0.0
                tope_max = 0.0125 if len(selecciones) >= 3 else 0.02
                stake_pct = min(tope_max, max(0.0, kelly_puro * 0.125))
                monto_apostar = bankroll_usuario * stake_pct

                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                k1.metric("Probabilidad Conjunta", f"{prob_total*100:.1f}%")
                k2.metric("Cuota Justa", f"{c_justa:.2f}")
                k3.metric("Cuota Mín (+5% EV)", f"{c_min_segura:.2f}")

                if ev >= 5.0:
                    st.success(
                        f"✅ **APOSTAR (+EV: {ev:+.2f}%)**\n\n"
                        f"💰 **Monto sugerido (1/8 Kelly):** `${monto_apostar:.2f} MXN` ({stake_pct*100:.2f}% de tu banca)"
                    )
                    if ev > 25.0:
                        st.warning("🚨 **Alerta (+25% EV):** Revisa alineaciones o lesiones de última hora antes de meterla.")
                elif ev > 0:
                    st.warning(
                        f"⚠️ **ZONA GRIS (EV: {ev:+.2f}%)**\n\n"
                        f"Tiene valor positivo pero es menor al colchón de seguridad de **+5.0% EV** (pide cuota >= `{c_min_segura:.2f}`)."
                    )
                else:
                    st.error(
                        f"❌ **NO APOSTAR (EV: {ev:+.2f}%)**\n\n"
                        f"Draftea te paga `{cuota_draftea:.2f}`, pero el riesgo real exige mínimo `{c_min_segura:.2f}`."
                    )


# =====================================================================
# PUNTO DE ENTRADA PRINCIPAL DE LA APLICACIÓN WEB
# =====================================================================
st.title("🏆 QuantBet Pro V8 — Fútbol + NBA + NFL")
st.caption("Motor Econométrico Multi-Deporte: Marcadores Exactos, Historial BTTS, Forma específica Local/Visita y Parlays SGP")

with st.sidebar:
    st.header("⚙️ Panel de Control")
    nombre_liga_sel = st.selectbox("1. Selecciona la Competición:", list(LIGAS_DISPONIBLES.keys()))
    cod_liga_sel, tipo_dep_sel = LIGAS_DISPONIBLES[nombre_liga_sel]
    banca_sel = st.number_input("💰 Tu Banca Total ($ MXN):", min_value=100.0, value=5000.0, step=100.0)

    st.divider()
    st.subheader("🚑 Ajuste por Bajas / Lesiones")
    cast_loc_sel = st.slider("Castigo ofensivo Local (%)", 0, 40, 0)
    cast_vis_sel = st.slider("Castigo ofensivo Visita (%)", 0, 40, 0)

if tipo_dep_sel in ("BASKETBALL", "FOOTBALL"):
    ejecutar_interfaz_nba_nfl(cod_liga_sel, nombre_liga_sel, banca_sel, cast_loc_sel, cast_vis_sel)
else:
    ejecutar_interfaz_futbol(cod_liga_sel, nombre_liga_sel, banca_sel, cast_loc_sel, cast_vis_sel)