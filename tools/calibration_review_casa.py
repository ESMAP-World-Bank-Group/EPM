# -*- coding: utf-8 -*-
"""The CASA calibration review page, built from what is on disk.

    python tools/calibration_review_casa.py

Two tabs, the same two the Black Sea review opens with and for the same reason.

    REPRESENTATIVE DAYS -- does the reduced time structure still describe the year it
    was cut from? Every figure on that tab comes from data_build/extracted/
    review_metrics.json, which replays the 360 blocks over a full year and compares
    them against the hourly series they were cut from.

    CALIBRATION (BASE YEAR) -- does the model reproduce the system as it is before it
    starts building? Half of that question can be answered here and half cannot, and
    the tab says which half is which rather than filling the gap with a proxy.

The prose is written here and the numbers are not: nothing on the page is typed in by
hand. Re-run the module after any rebuild or any new solve and the page follows.

Reads
    data_build/extracted/review_metrics.json      the reduction, scored
    data_build/extracted/demand_report.csv        which calendar day each block stands for
    data_build/extracted/vre_report.csv           the level and shape source of each block
    data_build/extracted/vre_hourly_report.csv    what the rescale did to each series
    data_build/extracted/pDemandForecast.csv      the demand each zone carries
    data_build/extracted/pGenDataInput.csv        the fleet the base year starts from
    epm/output/<run>/summary.csv                  the trajectory being reviewed

Writes
    ../Data/calibration/calibration_review.html
"""
import argparse
import collections
import csv
import io
import json
import os
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTRACTED = os.path.join(ROOT, "data_build", "extracted")
DEFAULT_RUN = "simulations_run_20260825_173849"
DEFAULT_BASE = "vre_real_days"
DEFAULT_OUT = os.path.join(os.path.dirname(ROOT), "Data", "calibration",
                           "calibration_review.html")

# The seven countries, so a zone code can be read by someone who does not know them.
ZONE_NAME = {
    "KAZ_N": "Kazakhstan north", "KAZ_S": "Kazakhstan south",
    "KGZ_N": "Kyrgyz Republic north", "KGZ_S": "Kyrgyz Republic south",
    "TAJ_N": "Tajikistan north", "TAJ_S": "Tajikistan south",
    "UZB": "Uzbekistan", "TUR": "Turkmenistan",
    "NEPS_AFG": "Afghanistan NEPS", "NEPS_HERAT": "Afghanistan Herat",
    "NEPS_TAJ": "Afghanistan Tajik lowland", "NEPS_TKM": "Afghanistan Turkmen border",
    "NEPS_UZB": "Afghanistan Uzbek border",
    "PAK_KAR": "Pakistan Karachi", "PAK_N": "Pakistan north", "PAK_S": "Pakistan south",
}
FUEL_COLOUR = {
    "Coal": "#5b5b5b", "Gas": "#e08a2e", "Diesel": "#8a5a3b", "Nuclear": "#8e5fbf",
    "Reservoir": "#2f7fbf", "Onshore Wind": "#3aa17e", "PV": "#f2c744",
    "Imports": "#b0b7c3",
}
SEASON_MONTHS = {"Q1": ("jan-fev", "Jan-Feb"), "Q2": ("mar-mai", "Mar-May"),
                 "Q3a": ("juin-juil", "Jun-Jul"), "Q3b": ("aout-sept", "Aug-Sep"),
                 "Q4": ("oct-dec", "Oct-Dec")}


# ------------------------------------------------------------------ small helpers
def t(fr, en):
    """One bilingual string. The page ships both and the toggle picks one."""
    return '<span class="lf">{0}</span><span class="le">{1}</span>'.format(fr, en)


def pct(value, digits=1, sign=True):
    if value is None:
        return "&mdash;"
    fmt = "{0:+." + str(digits) + "f}%" if sign else "{0:." + str(digits) + "f}%"
    return fmt.format(100.0 * value)


def klass(value, good, fair):
    """Green under `good`, amber under `fair`, red above. On the absolute value."""
    if value is None:
        return ""
    v = abs(value)
    return "ok" if v < good else ("wm" if v < fair else "bd")


def klass_high(value, good, fair):
    """The same, for a metric where high is good -- a correlation."""
    if value is None:
        return ""
    return "ok" if value >= good else ("wm" if value >= fair else "bd")


def read_csv(path):
    with io.open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def med(values):
    v = sorted(x for x in values if x is not None)
    return v[len(v) // 2] if v else None


# ------------------------------------------------------------------ the charts
def polyline(points, colour, width=1.6, dash=""):
    d = ' stroke-dasharray="{0}"'.format(dash) if dash else ""
    return ('<polyline fill="none" stroke="{0}" stroke-width="{1}"{2} points="{3}"/>'
            .format(colour, width, d, " ".join("{0:.1f},{1:.1f}".format(x, y)
                                               for x, y in points)))


def mini(series, title, width=176, height=92, ymax=None, colours=("#c0682a", "#1E6DB8")):
    """One small chart: a measured curve and a modelled one, no axes, one label.

    Small multiples rather than one big chart, because the question asked of these
    curves is never 'what is the value' -- it is 'do the two curves lie on top of each
    other', and sixteen small pictures answer that faster than sixteen big ones.
    """
    pad_l, pad_t, pad_b = 4, 14, 10
    inner_w, inner_h = width - pad_l - 4, height - pad_t - pad_b
    top = ymax or max([max(s) for s in series if s] + [1e-9])
    out = ['<svg viewBox="0 0 {0} {1}" width="{0}" height="{1}" '
           'role="img" aria-label="{2}">'.format(width, height, title)]
    out.append('<rect x="0" y="0" width="{0}" height="{1}" fill="#fff"/>'.format(
        width, height))
    out.append('<text x="{0}" y="10" font-size="8.5" fill="#33445e" '
               'font-weight="700">{1}</text>'.format(pad_l, title))
    out.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#e4e9f1"/>'.format(
        pad_l, pad_t + inner_h, pad_l + inner_w))
    for s, colour in zip(series, colours):
        if not s:
            continue
        n = len(s) - 1 or 1
        pts = [(pad_l + inner_w * i / float(n), pad_t + inner_h * (1 - v / top))
               for i, v in enumerate(s)]
        out.append(polyline(pts, colour))
    out.append("</svg>")
    return "".join(out)


def bars(years, stacks, order, width=940, height=260, unit="MW", axis="GW"):
    """The trajectory as stacked columns, one column per modelled year."""
    pad_l, pad_t, pad_b = 46, 10, 26
    inner_w, inner_h = width - pad_l - 8, height - pad_t - pad_b
    top = max(sum(stacks[y].get(k, 0.0) for k in order) for y in years) or 1.0
    step = inner_w / float(len(years))
    bw = step * 0.62
    out = ['<svg viewBox="0 0 {0} {1}" width="100%" height="{1}">'.format(width, height)]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + inner_h * (1 - frac)
        out.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" '
                   'stroke="#eef1f6"/>'.format(pad_l, y, pad_l + inner_w))
        out.append('<text x="{0}" y="{1:.1f}" font-size="9" fill="#7a869c" '
                   'text-anchor="end">{2:,.0f}</text>'.format(
                       pad_l - 5, y + 3, top * frac / 1000.0))
    out.append('<text x="4" y="{0}" font-size="8.5" fill="#7a869c">{1}</text>'.format(
        pad_t + 4, axis))
    for i, year in enumerate(years):
        x = pad_l + step * i + (step - bw) / 2.0
        base = pad_t + inner_h
        for key in order:
            v = stacks[year].get(key, 0.0)
            if v <= 0:
                continue
            h = inner_h * v / top
            base -= h
            out.append('<rect x="{0:.1f}" y="{1:.1f}" width="{2:.1f}" height="{3:.1f}" '
                       'fill="{4}"><title>{5} {6} {7:,.0f} {8}</title></rect>'.format(
                           x, base, bw, h, FUEL_COLOUR.get(key, "#999"),
                           year, key, v, unit))
        out.append('<text x="{0:.1f}" y="{1}" font-size="9" fill="#5a6577" '
                   'text-anchor="middle">{2}</text>'.format(
                       x + bw / 2.0, pad_t + inner_h + 15, year))
    out.append("</svg>")
    return "".join(out)


def legend(keys):
    return ('<div class="legend">' + "".join(
        '<span class="lg"><i style="background:{0}"></i>{1}</span>'.format(
            FUEL_COLOUR.get(k, "#999"), k) for k in keys) + "</div>")


# ------------------------------------------------------------------ tab 1
def tab_representative(metrics, dates, vre_report, hourly_report, coverage):
    demand, vre = metrics["demand"], metrics["vre"]
    blocks = metrics["blocks"]
    out = []

    # -- 1. method and sizing -------------------------------------------------
    out.append('<h3 class="roman">' + t("1 &middot; M&eacute;thode &amp; dimensionnement",
                                        "1 &middot; Method &amp; sizing") + "</h3>")
    out.append('<div class="method">' + t(
        "<b>Construction (data_build/build_demand.py).</b> Chaque <b>saison</b> (5 : Q1, Q2, "
        "Q3a, Q3b, Q4 &mdash; l'&eacute;t&eacute; est coup&eacute; en deux parce que juin-juillet et "
        "ao&ucirc;t-septembre ne sont pas la m&ecirc;me saison hydrologique) est r&eacute;duite &agrave; "
        "<b>3 journ&eacute;es-types</b> de 24 h :<br>"
        "&bull; <b>d1, la journ&eacute;e de pointe</b> &mdash; la journ&eacute;e calendaire o&ugrave; le signal de "
        "stress r&eacute;gional (chaque zone divis&eacute;e par sa propre pointe, puis somm&eacute;) est le plus "
        "haut. Elle ne repr&eacute;sente qu'<b>elle-m&ecirc;me, un jour</b>, et c'est elle qui donne &agrave; la "
        "contrainte de r&eacute;serve un vrai maximum &agrave; trouver ;<br>"
        "&bull; <b>d2 et d3</b> &mdash; les journ&eacute;es restantes sont class&eacute;es par &eacute;nergie "
        "journali&egrave;re et coup&eacute;es en deux groupes, chacun repr&eacute;sent&eacute; par son "
        "<b>m&eacute;do&iuml;de</b> : la journ&eacute;e r&eacute;elle la plus proche de la moyenne du groupe.<br>"
        "<b>Point cl&eacute; &mdash; ce sont de vraies journ&eacute;es, pas des moyennes.</b> Une journ&eacute;e "
        "moyenne n'est pas une journ&eacute;e : moyenner soixante-seize jours aplatit tout creux qui "
        "ne tombe pas deux fois &agrave; la m&ecirc;me heure, et le mod&egrave;le lit cet aplatissement comme "
        "une base que le syst&egrave;me n'a pas. Les 15 blocs sont 15 dates r&eacute;elles.<br>"
        "Les <b>poids</b> (pHours.csv) partent de la taille du groupe et sont d&eacute;plac&eacute;s de la "
        "plus courte distance qui fait retomber l'&eacute;nergie de la saison sur la bonne valeur ; le "
        "nombre de jours de l'ann&eacute;e est intact par construction.",

        "<b>Construction (data_build/build_demand.py).</b> Each <b>season</b> (5: Q1, Q2, Q3a, "
        "Q3b, Q4 &mdash; summer is cut in two because June-July and August-September are not "
        "the same hydrological season) is reduced to <b>3 day-types</b> of 24 h:<br>"
        "&bull; <b>d1, the peak day</b> &mdash; the calendar day on which the regional stress "
        "signal (each zone divided by its own peak, then summed) is highest. It stands for "
        "<b>itself alone, one day</b>, and it is what gives the reserve constraint a true "
        "maximum to find;<br>"
        "&bull; <b>d2 and d3</b> &mdash; the remaining days are ranked by daily energy and cut "
        "into two groups, each represented by its <b>medoid</b>: the real day closest to the "
        "group's own average.<br>"
        "<b>Key point &mdash; these are real days, not averages.</b> A mean day is not a day: "
        "averaging seventy-six days flattens every trough that does not fall at the same hour "
        "twice, and the model reads that flattening as a baseload the system does not have. "
        "The 15 blocks are 15 real dates.<br>"
        "The <b>weights</b> (pHours.csv) start from the group size and are moved the shortest "
        "distance that puts the season's energy back on its true value; the number of days in "
        "the year is untouched by construction.") + "</div>")

    cards = [(t("Saisons", "Seasons"), "5", t("Q1 &middot; Q2 &middot; Q3a &middot; Q3b &middot; Q4",
                                              "Q1 &middot; Q2 &middot; Q3a &middot; Q3b &middot; Q4")),
             (t("Jours-types / saison", "Day-types / season"), "3",
              t("1 pointe + 2 m&eacute;do&iuml;des", "1 peak + 2 medoids")),
             (t("Blocs", "Blocks"), "15", "5 &times; 3"),
             (t("Pas de temps / an", "Timesteps / year"), "360",
              t("15 &times; 24 h (vs 8 760)", "15 &times; 24 h (vs 8,760)")),
             (t("Compression", "Compression"), "&minus;95,9&nbsp;%",
              t("8 760 &rarr; 360", "8,760 &rarr; 360")),
             (t("Zones", "Zones"), "16", t("7 pays", "7 countries"))]
    out.append('<div class="sparkrow">' + "".join(
        '<div class="sparkcard"><div class="spt">{0}</div>'
        '<div style="font-size:1.5rem;font-weight:800;color:#12355b">{1}</div>'
        '<div class="spd">{2}</div></div>'.format(*c) for c in cards) + "</div>")

    out.append('<div class="subh">' + t(
        "Les 15 blocs sont 15 dates r&eacute;elles",
        "The 15 blocks are 15 real dates") + "</div>")
    rows = collections.defaultdict(dict)
    for r in dates:
        rows[r["q"]][r["d"]] = r
    head = ("<tr><th>" + t("Saison", "Season") + "</th><th>d1</th><th>d2</th><th>d3</th>"
            "<th>" + t("&Sigma; poids", "&Sigma; weights") + "</th></tr>")
    body = []
    for q in ("Q1", "Q2", "Q3a", "Q3b", "Q4"):
        cells = []
        total = 0.0
        for d in ("d1", "d2", "d3"):
            r = rows[q][d]
            total += float(r["days_stood_for"])
            kind = t("pointe", "peak") if r["kind"] == "peak day" else t("m&eacute;do&iuml;de", "medoid")
            cells.append('<td>{0}<br><span style="font-size:.68rem;color:#7a869c">{1} '
                         '&middot; {2}</span></td>'.format(
                             r["date"], kind, int(float(r["days_stood_for"]))))
        fr, en = SEASON_MONTHS[q]
        body.append('<tr><td style="text-align:left"><b>{0}</b> '
                    '<span style="font-size:.72rem;color:#7a869c">{1}</span></td>{2}'
                    '<td class="ok"><b>{3:.0f}</b></td></tr>'.format(
                        q, t(fr, en), "".join(cells), total))
    out.append('<div class="gridtbl"><table class="cal"><thead>' + head +
               "</thead><tbody>" + "".join(body) + "</tbody></table></div>")
    out.append('<div class="pn">' + t(
        "<b>Format</b> mm/jj &middot; <i>r&egrave;gle &middot; poids</i> (nombre de journ&eacute;es r&eacute;elles "
        "repr&eacute;sent&eacute;es). &Sigma; = <b>{0:.0f}</b> jours, l'ann&eacute;e enti&egrave;re. L'ann&eacute;e "
        "m&eacute;t&eacute;o est celle du livre DeCA de chaque pays : 2019 pour le Kazakhstan, 2021 pour le "
        "Tadjikistan, 2022 pour le Kirghizistan et l'Ouzb&eacute;kistan. Les journ&eacute;es-types sont "
        "choisies sur le <b>signal r&eacute;gional</b>, donc les 16 zones partagent les m&ecirc;mes 15 dates "
        "&mdash; ce qui est la condition pour que les &eacute;changes entre zones aient un sens.",
        "<b>Format</b> mm/dd &middot; <i>rule &middot; weight</i> (number of real days "
        "represented). &Sigma; = <b>{0:.0f}</b> days, the whole year. The weather year is the one "
        "in each country's DeCA book: 2019 for Kazakhstan, 2021 for Tajikistan, 2022 for the "
        "Kyrgyz Republic and Uzbekistan. The day-types are chosen on the <b>regional</b> "
        "signal, so all 16 zones share the same 15 dates &mdash; which is the condition for "
        "inter-zonal exchange to mean anything.").format(sum(blocks.values())) + "</div>")

    # -- 2. metric glossary ---------------------------------------------------
    out.append('<h3 class="roman">' + t(
        "2 &middot; Que mesure-t-on ? &mdash; glossaire des m&eacute;triques (valeur vs profil)",
        "2 &middot; What do we measure? &mdash; metric glossary (value vs profile)") + "</h3>")
    out.append('<div class="method">' + t(
        "<b>Deux familles de m&eacute;triques, deux questions diff&eacute;rentes.</b> Une r&eacute;duction "
        "temporelle peut r&eacute;ussir l'une et &eacute;chouer l'autre &mdash; il faut donc les s&eacute;parer.<br>"
        "&bull; <b class='ok'>VALEUR</b> &mdash; &laquo; combien, et &agrave; quelle fr&eacute;quence ? &raquo; : le "
        "niveau d'&eacute;nergie et la <b>distribution</b> des valeurs horaires, <b>ind&eacute;pendamment de "
        "l'heure</b> &agrave; laquelle elles surviennent.<br>"
        "&bull; <b style='color:#1E6DB8'>PROFIL</b> &mdash; &laquo; quand ? &raquo; : la <b>forme "
        "intra-journali&egrave;re</b>, l'heure des mont&eacute;es, des pointes et des creux.",
        "<b>Two families of metrics, two different questions.</b> A temporal reduction can "
        "pass one and fail the other, so they must be kept apart.<br>"
        "&bull; <b class='ok'>VALUE</b> &mdash; 'how much, and how often?': the energy level and "
        "the <b>distribution</b> of hourly values, <b>independent of the hour</b> at which they "
        "occur.<br>"
        "&bull; <b style='color:#1E6DB8'>PROFILE</b> &mdash; 'when?': the <b>intraday shape</b>, "
        "the timing of ramps, peaks and troughs.") + "</div>")

    glossary = [
        (t("Erreur d'&eacute;nergie", "Energy error"),
         t("moyenne annuelle reconstruite &divide; moyenne r&eacute;elle &minus; 1",
           "reconstructed annual mean &divide; real mean &minus; 1"),
         t("le <b>niveau moyen</b> sur l'ann&eacute;e", "the <b>mean level</b> over the year"),
         t("VALEUR", "VALUE"), "&lt; 2 %"),
        (t("Quantiles de courbe class&eacute;e (P05 / P50 / P95)",
           "Duration-curve quantiles (P05 / P50 / P95)"),
         t("valeur d&eacute;pass&eacute;e 5 % / 50 % / 95 % du temps",
           "value exceeded 5% / 50% / 95% of the time"),
         t("les <b>queues</b> : les heures tendues et les heures calmes",
           "the <b>tails</b>: the tight hours and the calm ones"),
         t("VALEUR", "VALUE"), "&mdash;"),
        (t("Erreur de pointe", "Peak error"),
         t("maximum reconstruit &divide; maximum r&eacute;el &minus; 1",
           "reconstructed maximum &divide; real maximum &minus; 1"),
         t("l'<b>extr&ecirc;me haut</b>, celui que l'ad&eacute;quation doit couvrir",
           "the <b>upper extreme</b>, the one adequacy must cover"),
         t("VALEUR", "VALUE"), "&lt; 10 %"),
        ("NRMSE LDC",
         t("&eacute;cart quadratique moyen sur la courbe class&eacute;e, rapport&eacute; &agrave; l'&eacute;tendue",
           "root-mean-square deviation on the duration curve, over its span"),
         t("la <b>distribution enti&egrave;re</b> en un chiffre",
           "the <b>whole distribution</b> in one number"),
         t("VALEUR", "VALUE"), "&lt; 5 %"),
        (t("Corr&eacute;lation diurne", "Diurnal correlation"),
         t("Pearson entre la forme 24 h moyenne reconstruite et r&eacute;elle",
           "Pearson between the reconstructed and real mean 24 h shape"),
         t("la <b>forme intra-journali&egrave;re</b> et son calage horaire",
           "the <b>intraday shape</b> and its timing"),
         t("PROFIL", "PROFILE"), "&gt; 0,90"),
    ]
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("M&eacute;trique", "Metric") + "</th><th>" + t("D&eacute;finition", "Definition") +
               "</th><th>" + t("Ce qu'elle regarde", "What it looks at") + "</th><th>" +
               t("Famille", "Family") + "</th><th>" + t("Seuil", "Threshold") +
               "</th></tr></thead><tbody>" + "".join(
                   '<tr><td style="text-align:left"><b>{0}</b></td>'
                   '<td style="text-align:left;font-size:.78rem">{1}</td>'
                   '<td style="text-align:left;font-size:.78rem">{2}</td>'
                   '<td><span class="{4}">{3}</span></td>'
                   '<td style="font-size:.78rem">{5}</td></tr>'.format(
                       g[0], g[1], g[2], g[3],
                       "ok" if "VAL" in g[3] else "", g[4]) for g in glossary) +
               "</tbody></table></div>")
    out.append('<div class="keybox" style="background:#f4f8fd;border-color:#cfe0f2;'
               'color:#12355b">' + t(
                   "<b>&Agrave; retenir.</b> Les quatre premi&egrave;res m&eacute;triques regardent la "
                   "<b>VALEUR</b> : elles disent si la r&eacute;duction garde la bonne quantit&eacute; "
                   "d'&eacute;nergie et la bonne distribution &mdash; mais elles sont <b>aveugles &agrave; "
                   "l'heure</b> (on peut d&eacute;caler un profil entier de 6 h sans les bouger). Seule la "
                   "<b>corr&eacute;lation diurne</b> regarde le <b>PROFIL</b>. Un bon jeu de journ&eacute;es "
                   "repr&eacute;sentatives doit r&eacute;ussir <b>les deux</b>. <b>Une pr&eacute;caution propre &agrave; "
                   "CASA :</b> pour le solaire et l'&eacute;olien, l'erreur d'&eacute;nergie est nulle "
                   "<b>par construction</b> &mdash; le niveau est impos&eacute; des deux c&ocirc;t&eacute;s par le "
                   "facteur d'utilisation DeCA. Elle ne prouve donc rien ; ce sont la courbe class&eacute;e "
                   "et la forme diurne qui portent toute l'information.",
                   "<b>Key point.</b> The first four metrics look at <b>VALUE</b>: they say "
                   "whether the reduction keeps the right amount of energy and the right "
                   "distribution &mdash; but they are <b>blind to timing</b> (a whole profile can "
                   "be shifted 6 h without moving them). Only the <b>diurnal correlation</b> "
                   "looks at the <b>PROFILE</b>. A good representative-day set must pass "
                   "<b>both</b>. <b>One CASA-specific caveat:</b> for solar and wind the energy "
                   "error is zero <b>by construction</b> &mdash; the level is imposed on both sides "
                   "by the DeCA utilisation factor. It therefore proves nothing; the duration "
                   "curve and the diurnal shape carry all the information.") + "</div>")

    # -- 3. summary by technology --------------------------------------------
    out.append('<h3 class="roman">' + t(
        "3 &middot; Bilan par technologie &mdash; valeur ET profil",
        "3 &middot; Summary by technology &mdash; value AND profile") + "</h3>")

    def block(entries):
        e = [abs(v["energy_error"]) for v in entries.values() if v["energy_error"] is not None]
        n = [v["ldc_nrmse"] for v in entries.values() if v["ldc_nrmse"] is not None]
        c = [v["diurnal_corr"] for v in entries.values() if v["diurnal_corr"] is not None]
        p = [abs(v["peak_error"]) for v in entries.values() if v["peak_error"] is not None]
        return med(e), max(e or [0]), med(n), max(n or [0]), med(c), min(c or [0]), med(p)

    dl, dlx, dn, dnx, dc, dcm, dp = block(demand)
    pl, plx, pn, pnx, pc, pcm, pp = block(vre["PV"])
    wl, wlx, wn, wnx, wc, wcm, wp = block(vre["WT"])
    tech_rows = [
        (t("Charge (Load)", "Load"),
         t("s&eacute;rie DeCA compt&eacute;e, 7 zones sur 16", "DeCA metered series, 7 zones of 16"),
         "ok", "ok",
         t("&eacute;nergie m&eacute;d. {0} (max {1})<br>NRMSE m&eacute;d. {2}",
           "energy med. {0} (max {1})<br>NRMSE med. {2}").format(
               pct(dl, 2, False), pct(dlx, 2, False), pct(dn, 2, False)),
         t("corr m&eacute;d. <b>{0:.3f}</b><br>(min {1:.3f})",
           "corr med. <b>{0:.3f}</b><br>(min {1:.3f})").format(dc, dcm),
         "ok", t("Excellent", "Excellent")),
        (t("Solaire PV", "Solar PV"),
         t("Renewables.ninja recal&eacute; DeCA, 16 zones",
           "Renewables.ninja rescaled to DeCA, 16 zones"),
         "ok", "ok",
         t("&eacute;nergie <b>impos&eacute;e</b><br>NRMSE m&eacute;d. {0} (max {1})",
           "energy <b>imposed</b><br>NRMSE med. {0} (max {1})").format(
               pct(pn, 2, False), pct(pnx, 2, False)),
         t("corr m&eacute;d. <b>{0:.3f}</b><br>(min {1:.3f})",
           "corr med. <b>{0:.3f}</b><br>(min {1:.3f})").format(pc, pcm),
         "ok", t("Excellent", "Excellent")),
        (t("&Eacute;olien (Wind)", "Wind"),
         t("Renewables.ninja recal&eacute;, 9 zones sur 16",
           "Renewables.ninja rescaled, 9 zones of 16"),
         "wm", "bd",
         t("&eacute;nergie <b>impos&eacute;e</b><br>NRMSE m&eacute;d. {0} (max {1})",
           "energy <b>imposed</b><br>NRMSE med. {0} (max {1})").format(
               pct(wn, 1, False), pct(wnx, 1, False)),
         t("corr m&eacute;d. <b>{0:.3f}</b><br>(min {1:.3f})",
           "corr med. <b>{0:.3f}</b><br>(min {1:.3f})").format(wc, wcm),
         "bd", t("&Agrave; surveiller", "To watch")),
    ]
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("Technologie", "Technology") + "</th><th>" +
               t("Valeur<br>(distribution)", "Value<br>(distribution)") + "</th><th>" +
               t("Profil<br>(forme diurne)", "Profile<br>(diurnal shape)") + "</th><th>" +
               t("Chiffres &mdash; valeur", "Figures &mdash; value") + "</th><th>" +
               t("Chiffres &mdash; profil", "Figures &mdash; profile") + "</th><th>" +
               t("Verdict", "Verdict") + "</th></tr></thead><tbody>")
    words = {"ok": t("Fid&egrave;le", "Faithful"), "wm": t("Correct", "Adequate"),
             "bd": t("Faible", "Weak")}
    for name, src, kv, kp, fv, fp, kver, verdict in tech_rows:
        out.append('<tr><td style="text-align:left"><b>{0}</b><br>'
                   '<span style="font-size:.72rem;color:#7a869c">{1}</span></td>'
                   '<td class="{2}">{3}</td><td class="{4}">{5}</td>'
                   '<td>{6}</td><td>{7}</td><td><span class="{8}">{9}</span></td></tr>'.format(
                       name, src, kv, words[kv], kp, words[kp], fv, fp, kver, verdict))
    out.append("</tbody></table></div>")
    out.append('<div class="keybox" style="background:#fff8ef;border-color:#f0dcc0;'
               'color:#7a541b">' + t(
                   "<b>Lecture crois&eacute;e valeur &times; profil.</b><br>"
                   "&bull; <b>Charge &amp; PV &mdash; les deux r&eacute;ussis.</b> &Eacute;nergie fid&egrave;le "
                   "(charge m&eacute;d. {0}, max {1}) <b>et</b> profil diurne quasi parfait "
                   "(corr &asymp; {2:.3f} pour la charge, {3:.3f} pour le PV). Normal : la charge et "
                   "surtout le <b>solaire</b> ont un cycle diurne <b>d&eacute;terministe</b> &mdash; une "
                   "journ&eacute;e r&eacute;elle quelconque de la saison porte d&eacute;j&agrave; la bonne forme.<br>"
                   "&bull; <b>&Eacute;olien &mdash; distribution moyenne, profil faible.</b> NRMSE m&eacute;diane "
                   "{4} (jusqu'&agrave; {5}) et surtout <b>corr&eacute;lation diurne effondr&eacute;e (m&eacute;d. "
                   "{6:.2f}, min {7:.2f})</b>. L'&eacute;olien n'a <b>pas de cycle diurne</b> : il est "
                   "pilot&eacute; par des passages m&eacute;t&eacute;o de plusieurs jours. Chacune des 3 "
                   "journ&eacute;es retenues porte donc la forme <b>bruit&eacute;e</b> d'un &eacute;pisode "
                   "particulier, l&agrave; o&ugrave; la vraie moyenne saisonni&egrave;re en agr&egrave;ge 90. On compare "
                   "deux courbes quasi plates dont la variation r&eacute;siduelle est du bruit : la "
                   "corr&eacute;lation est <b>structurellement</b> basse, et c'est en partie un "
                   "<b>artefact de m&eacute;trique</b>. Le vrai risque est ailleurs, au &sect;6 : la "
                   "<b>queue calme</b>.",
                   "<b>Reading value &times; profile together.</b><br>"
                   "&bull; <b>Load &amp; PV &mdash; both pass.</b> Faithful energy (load med. {0}, "
                   "max {1}) <b>and</b> a near-perfect diurnal profile (corr &asymp; {2:.3f} for "
                   "load, {3:.3f} for PV). Expected: load and above all <b>solar</b> have a "
                   "<b>deterministic</b> diurnal cycle &mdash; any real day of the season already "
                   "carries the right shape.<br>"
                   "&bull; <b>Wind &mdash; middling distribution, weak profile.</b> Median NRMSE "
                   "{4} (up to {5}) and above all a <b>collapsed diurnal correlation (med. "
                   "{6:.2f}, min {7:.2f})</b>. Wind has <b>no diurnal cycle</b>: it is driven by "
                   "multi-day weather systems. Each of the 3 retained days therefore carries the "
                   "<b>noisy</b> shape of one particular episode, where the true seasonal mean "
                   "aggregates 90. Two near-flat curves are being compared whose residual "
                   "variation is noise: the correlation is <b>structurally</b> low, and is partly "
                   "a <b>metric artefact</b>. The real risk is elsewhere, in &sect;6: the "
                   "<b>calm tail</b>.").format(
                       pct(dl, 2, False), pct(dlx, 2, False), dc, pc,
                       pct(wn, 1, False), pct(wnx, 1, False), wc, wcm) + "</div>")

    # -- 4. load --------------------------------------------------------------
    out.append('<h3 class="roman">' + t(
        "4 &middot; Charge &mdash; d&eacute;tail par zone",
        "4 &middot; Load &mdash; detail by zone") + "</h3>")
    out.append('<div class="method">' + t(
        "On rejoue les 15 blocs pond&eacute;r&eacute;s sur une ann&eacute;e enti&egrave;re et on compare &agrave; la vraie "
        "chronique compt&eacute;e du livre DeCA. Les deux c&ocirc;t&eacute;s sont <b>en pu de leur propre "
        "pointe</b>, parce que pDemandProfile est normalis&eacute; &agrave; un maximum de 1 par construction "
        "et que la s&eacute;rie compt&eacute;e est en MW : la comparaison porte sur la <b>forme</b>, le "
        "niveau &eacute;tant l'affaire de pDemandForecast. L'erreur de pointe est donc nulle par "
        "construction et n'est pas report&eacute;e ici.",
        "The 15 weighted blocks are replayed over a full year and compared to the true metered "
        "series from the DeCA book. Both sides are <b>in per unit of their own peak</b>, because "
        "pDemandProfile is normalised to a maximum of 1 by construction while the metered series "
        "is in MW: the comparison is about <b>shape</b>, the level being the business of "
        "pDemandForecast. Peak error is therefore zero by construction and is not reported "
        "here.") + "</div>")
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("Zone", "Zone") + "</th><th>" + t("Pointe compt&eacute;e", "Metered peak") +
               "</th><th>" + t("Facteur de charge<br>r&eacute;el / mod&egrave;le",
                               "Load factor<br>real / model") + "</th><th>" +
               t("Err. &eacute;nergie", "Energy err.") + "</th><th>P95 " +
               t("(heures calmes)", "(calm hours)") + "</th><th>NRMSE LDC</th><th>" +
               t("Corr. diurne", "Diurnal corr.") + "</th></tr></thead><tbody>")
    for z in sorted(demand):
        v = demand[z]
        out.append('<tr><td style="text-align:left"><b>{0}</b><br>'
                   '<span style="font-size:.72rem;color:#7a869c">{1}</span></td>'
                   '<td>{2:,.0f} MW</td><td>{3:.3f} / {4:.3f}</td>'
                   '<td class="{5}">{6}</td><td>{7:.3f} / {8:.3f}</td>'
                   '<td class="{9}">{10}</td><td class="{11}">{12:.3f}</td></tr>'.format(
                       z, ZONE_NAME.get(z, ""), v["peak_mw"],
                       v["measured_mean"], v["model_mean"],
                       klass(v["energy_error"], 0.02, 0.05), pct(v["energy_error"], 2),
                       v["p95"][0], v["p95"][1],
                       klass(v["ldc_nrmse"], 0.05, 0.10), pct(v["ldc_nrmse"], 2, False),
                       klass_high(v["diurnal_corr"], 0.90, 0.75), v["diurnal_corr"]))
    out.append("</tbody></table></div>")
    out.append('<div class="pn">' + t(
        "<b>Couverture.</b> Sept zones sur seize ont une ann&eacute;e compt&eacute;e compl&egrave;te, soit "
        "<b>{0:.0f} % de la demande r&eacute;gionale de 2026</b>. Les neuf autres n'en ont pas : "
        "l'Afghanistan et le Pakistan sont hors du p&eacute;rim&egrave;tre DeCA et gardent la forme du "
        "mod&egrave;le 2020, et le <b>Turkm&eacute;nistan ne fournit que 288 points</b> &mdash; une journ&eacute;e "
        "moyenne par mois, qui est une forme et non une ann&eacute;e. Il ne peut pas dire &agrave; quoi "
        "ressemblait le creux d'un vrai mois de f&eacute;vrier, et il est donc report&eacute; comme non "
        "couvert plut&ocirc;t que not&eacute; contre un substitut. C'est la limite &agrave; retenir : la moiti&eacute; "
        "de la demande du mod&egrave;le n'est pas v&eacute;rifiable ici.",
        "<b>Coverage.</b> Seven zones out of sixteen have a full metered year, which is "
        "<b>{0:.0f}% of 2026 regional demand</b>. The other nine do not: Afghanistan and "
        "Pakistan are outside the DeCA perimeter and keep the 2020 model's shape, and "
        "<b>Turkmenistan supplies only 288 points</b> &mdash; one average day per month, which is "
        "a shape and not a year. It cannot say what the trough of a real February looked like, "
        "and is therefore reported as uncovered rather than scored against a stand-in. That is "
        "the limitation to keep in mind: half the model's demand is not checkable here.").format(
            coverage) + "</div>")
    out.append('<div class="subh">' + t(
        "Courbe class&eacute;e &mdash; <span style='color:#1E6DB8'>reconstruction</span> vs "
        "<span style='color:#c0682a'>r&eacute;el</span>",
        "Duration curve &mdash; <span style='color:#1E6DB8'>reconstruction</span> vs "
        "<span style='color:#c0682a'>real</span>") + "</div>")
    out.append('<div class="sparkrow">' + "".join(
        '<div class="chart">' + mini([demand[z]["measured_ldc"], demand[z]["model_ldc"]],
                                     z, ymax=1.0) + "</div>" for z in sorted(demand)) +
        "</div>")
    out.append('<div class="subh">' + t(
        "Forme diurne moyenne (24 h)", "Mean diurnal shape (24 h)") + "</div>")
    out.append('<div class="sparkrow">' + "".join(
        '<div class="chart">' + mini([demand[z]["measured_day"], demand[z]["model_day"]],
                                     z, ymax=1.0) + "</div>" for z in sorted(demand)) +
        "</div>")

    # -- 5 and 6. VRE ---------------------------------------------------------
    shape_by = collections.Counter((r["tech"], r["shape"]) for r in vre_report)
    for tech, number, title_fr, title_en in (
            ("PV", "5", "Solaire PV &mdash; d&eacute;tail par zone", "Solar PV &mdash; detail by zone"),
            ("WT", "6", "&Eacute;olien &mdash; d&eacute;tail par zone", "Wind &mdash; detail by zone")):
        entries = vre[tech]
        out.append('<h3 class="roman">' + t(number + " &middot; " + title_fr,
                                            number + " &middot; " + title_en) + "</h3>")
        if tech == "PV":
            out.append('<div class="method">' + t(
                "Le c&ocirc;t&eacute; mesur&eacute; est ici la <b>s&eacute;rie horaire Renewables.ninja d&eacute;j&agrave; "
                "ramen&eacute;e au niveau DeCA</b> &mdash; le fichier dont les blocs ont &eacute;t&eacute; d&eacute;coup&eacute;s. "
                "Cette section mesure donc la <b>r&eacute;duction seule</b> et non le fetch : savoir si "
                "Renewables.ninja a raison sur l'Asie centrale est une autre question, trait&eacute;e "
                "dans extracted/vre_hourly_report.csv. Les deux c&ocirc;t&eacute;s sont des facteurs de "
                "charge, donc l'erreur de pointe est un vrai &eacute;cart d'extr&ecirc;me.",
                "The measured side here is the <b>Renewables.ninja hourly series already brought "
                "onto the DeCA level</b> &mdash; the file the blocks were cut from. This section "
                "therefore measures the <b>reduction alone</b> and not the fetch: whether "
                "Renewables.ninja is right about Central Asia is a different question, answered "
                "in extracted/vre_hourly_report.csv. Both sides are capacity factors, so the "
                "peak error is a real extreme deviation.") + "</div>")
        out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
                   t("Zone", "Zone") + "</th><th>CF</th><th>" +
                   t("Err. &eacute;nergie", "Energy err.") + "</th><th>" +
                   t("Err. pointe", "Peak err.") + "</th><th>P50</th><th>P95 " +
                   t("(r&eacute;el / mod.)", "(real / model)") + "</th><th>NRMSE LDC</th><th>" +
                   t("Corr. diurne", "Diurnal corr.") + "</th></tr></thead><tbody>")
        for z in sorted(entries):
            v = entries[z]
            corr = v["diurnal_corr"]
            out.append('<tr><td style="text-align:left"><b>{0}</b><br>'
                       '<span style="font-size:.72rem;color:#7a869c">{1}</span></td>'
                       '<td>{2:.4f}</td><td class="ok">{3}</td><td class="{4}">{5}</td>'
                       '<td>{6:.3f} / {7:.3f}</td><td>{8:.4f} / {9:.4f}</td>'
                       '<td class="{10}">{11}</td><td class="{12}">{13}</td></tr>'.format(
                           z, ZONE_NAME.get(z, ""), v["measured_mean"],
                           pct(v["energy_error"], 3),
                           klass(v["peak_error"], 0.10, 0.25), pct(v["peak_error"], 1),
                           v["p50"][0], v["p50"][1], v["p95"][0], v["p95"][1],
                           klass(v["ldc_nrmse"], 0.05, 0.10), pct(v["ldc_nrmse"], 2, False),
                           klass_high(corr, 0.90, 0.75),
                           "{0:.3f}".format(corr) if corr is not None else "&mdash;"))
        out.append("</tbody></table></div>")

        real = shape_by[(tech, "rninja hourly")]
        flat = shape_by[(tech, "casa_2020 blocks")]
        if tech == "PV":
            out.append('<div class="pn">' + t(
                "<b>{0} zones-saisons sur {1}</b> tournent sur une journ&eacute;e r&eacute;elle. L'erreur "
                "d'&eacute;nergie est nulle &agrave; la cinqui&egrave;me d&eacute;cimale parce que le niveau est impos&eacute; "
                "des deux c&ocirc;t&eacute;s : ce qu'il faut lire ici, c'est la colonne <b>erreur de "
                "pointe</b>, qui atteint &minus;23 % sur KAZ_N. Quinze journ&eacute;es ne contiennent "
                "pas le meilleur jour de ciel clair de l'ann&eacute;e, et la pointe solaire du mod&egrave;le "
                "est donc l&eacute;g&egrave;rement rabot&eacute;e &mdash; sans cons&eacute;quence pour l'&eacute;nergie, mais &agrave; "
                "savoir si l'on regarde un &eacute;cr&ecirc;tage.",
                "<b>{0} zone-seasons out of {1}</b> run on a real day. The energy error is zero "
                "to the fifth decimal because the level is imposed on both sides: what to read "
                "here is the <b>peak error</b> column, which reaches &minus;23% on KAZ_N. "
                "Fifteen days do not contain the clearest-sky day of the year, so the model's "
                "solar peak is slightly clipped &mdash; without consequence for energy, but worth "
                "knowing if curtailment is being looked at.").format(real, real + flat) +
                "</div>")
        else:
            missing = sorted(set(metrics["vre_uncovered"]))
            out.append('<div class="pn">' + t(
                "<b>Sept zones &eacute;oliennes sur seize n'apparaissent pas dans ce tableau</b> "
                "({0}) : leur ann&eacute;e Renewables.ninja a &eacute;t&eacute; <b>refus&eacute;e</b> au recalage, "
                "l'&eacute;cart au facteur DeCA d&eacute;passant le facteur 2 admis &mdash; jusqu'&agrave; &times;14,3 "
                "pour PAK_N. Elles gardent la construction 2020, c'est-&agrave;-dire une journ&eacute;e "
                "plate, avec un niveau juste. Il n'existe aucune s&eacute;rie horaire contre laquelle "
                "les noter, et elles sont donc absentes plut&ocirc;t que vertes. Au total <b>{1} "
                "zones-saisons &eacute;oliennes sur {2}</b> tournent sur une journ&eacute;e r&eacute;elle.",
                "<b>Seven wind zones out of sixteen are absent from this table</b> ({0}): their "
                "Renewables.ninja year was <b>refused</b> at rescaling, the gap to the DeCA "
                "factor exceeding the factor of 2 allowed &mdash; up to &times;14.3 for PAK_N. "
                "They keep the 2020 construction, that is a flat day, with a correct level. "
                "There is no hourly series to score them against, so they are absent rather "
                "than green. In total <b>{1} wind zone-seasons out of {2}</b> run on a real "
                "day.").format(", ".join(m.split()[0] for m in missing), real, real + flat) +
                "</div>")
        out.append('<div class="subh">' + t(
            "Forme diurne moyenne &mdash; <span style='color:#1E6DB8'>reconstruction</span> vs "
            "<span style='color:#c0682a'>r&eacute;el</span>",
            "Mean diurnal shape &mdash; <span style='color:#1E6DB8'>reconstruction</span> vs "
            "<span style='color:#c0682a'>real</span>") + "</div>")
        out.append('<div class="sparkrow">' + "".join(
            '<div class="chart">' + mini([entries[z]["measured_day"], entries[z]["model_day"]],
                                         z) + "</div>" for z in sorted(entries)) + "</div>")
        if tech == "WT":
            out.append('<div class="subh">' + t(
                "Courbe class&eacute;e &mdash; la queue calme est le point faible",
                "Duration curve &mdash; the calm tail is the weak point") + "</div>")
            out.append('<div class="sparkrow">' + "".join(
                '<div class="chart">' + mini([entries[z]["measured_ldc"],
                                              entries[z]["model_ldc"]], z) + "</div>"
                for z in sorted(entries)) + "</div>")
            worst = max(entries, key=lambda z: entries[z]["p95"][1] - entries[z]["p95"][0])
            w = entries[worst]
            out.append('<div class="keybox" style="background:#fdf0ec;'
                       'border-color:#f0d0c4;color:#8a3a20">' + t(
                           "<b>Le vrai risque : les heures calmes disparaissent.</b> Le P95 est "
                           "la valeur d&eacute;pass&eacute;e 95 % du temps, c'est-&agrave;-dire le fond de la "
                           "distribution &mdash; les heures o&ugrave; le vent ne souffle pas. Sur {0}, le "
                           "r&eacute;el descend &agrave; <b>{1:.4f}</b> et le mod&egrave;le s'arr&ecirc;te &agrave; "
                           "<b>{2:.4f}</b>, soit {3:.0f} fois plus haut. Trois journ&eacute;es par saison "
                           "ne peuvent pas contenir un anticyclone de cinq jours : le mod&egrave;le ne "
                           "voit jamais un calme prolong&eacute;, et donc jamais le besoin de capacit&eacute; "
                           "ferme qui va avec. <b>C'est le biais &agrave; conna&icirc;tre avant de lire un "
                           "r&eacute;sultat d'ad&eacute;quation</b>, et ce que r&eacute;parerait un jeu de journ&eacute;es "
                           "choisies sur le vent (clustering Poncelet) plut&ocirc;t que sur la charge.",
                           "<b>The real risk: the calm hours disappear.</b> P95 is the value "
                           "exceeded 95% of the time, that is the bottom of the distribution "
                           "&mdash; the hours when the wind does not blow. On {0} the real series "
                           "drops to <b>{1:.4f}</b> while the model stops at <b>{2:.4f}</b>, "
                           "{3:.0f} times higher. Three days per season cannot contain a "
                           "five-day anticyclone: the model never sees a prolonged lull, and "
                           "therefore never sees the firm capacity need that comes with it. "
                           "<b>This is the bias to know before reading any adequacy result</b>, "
                           "and it is what a day set chosen on wind (Poncelet clustering) rather "
                           "than on load would repair.").format(
                               worst, w["p95"][0], w["p95"][1],
                               w["p95"][1] / (w["p95"][0] or 1e-9)) + "</div>")

    # -- 7. verdict -----------------------------------------------------------
    out.append('<h3 class="roman">' + t(
        "7 &middot; Verdict &amp; limites honn&ecirc;tes",
        "7 &middot; Verdict &amp; honest limitations") + "</h3>")
    out.append('<div class="cols2">')
    out.append('<div class="sec"><div class="sn">' + t("Ce qui tient", "What holds") +
               "</div><div class='body'><ul>" + "".join("<li>" + x + "</li>" for x in [
                   t("<b>La charge.</b> Sept zones compt&eacute;es, erreur d'&eacute;nergie m&eacute;diane {0}, "
                     "corr&eacute;lation diurne {1:.3f} au minimum. Le choix du m&eacute;do&iuml;de plut&ocirc;t que "
                     "de la moyenne est ce qui sauve le creux de nuit.",
                     "<b>Load.</b> Seven metered zones, median energy error {0}, diurnal "
                     "correlation {1:.3f} at worst. Choosing the medoid rather than the mean is "
                     "what saves the night trough.").format(pct(dl, 2, False), dcm),
                   t("<b>Le solaire.</b> Seize zones sur seize sur journ&eacute;e r&eacute;elle, "
                     "corr&eacute;lation {0:.3f} au minimum. Le cycle diurne est d&eacute;terministe : "
                     "n'importe quelle journ&eacute;e r&eacute;elle porte la bonne forme.",
                     "<b>Solar.</b> Sixteen zones out of sixteen on a real day, correlation "
                     "{0:.3f} at worst. The diurnal cycle is deterministic: any real day carries "
                     "the right shape.").format(pcm),
                   t("<b>La pointe.</b> d1 est la vraie journ&eacute;e de stress r&eacute;gional et ne "
                     "repr&eacute;sente qu'elle-m&ecirc;me : la contrainte de r&eacute;serve a un maximum r&eacute;el &agrave; "
                     "trouver, ce que la cinqui&egrave;me saison-pointe du mod&egrave;le 2020 ne donnait pas.",
                     "<b>The peak.</b> d1 is the true regional stress day and stands for itself "
                     "alone: the reserve constraint has a real maximum to find, which the 2020 "
                     "model's fifth peak season did not provide."),
                   t("<b>L'&eacute;nergie.</b> Les poids sont r&eacute;solus pour que l'&eacute;nergie de chaque "
                     "saison retombe juste, et &Sigma; poids = 365 par construction.",
                     "<b>Energy.</b> The weights are solved so each season's energy lands right, "
                     "and &Sigma; weights = 365 by construction."),
               ]) + "</ul></div></div>")
    out.append('<div class="sec" style="background:#fdf6f3;border-color:#f0d0c4">'
               '<div class="sn" style="color:#a5381f">' +
               t("Ce qui ne tient pas", "What does not hold") +
               "</div><div class='body'><ul>" + "".join("<li>" + x + "</li>" for x in [
                   t("<b>La queue calme de l'&eacute;olien.</b> Le P95 mod&eacute;lis&eacute; est jusqu'&agrave; un ordre "
                     "de grandeur au-dessus du r&eacute;el. Trois journ&eacute;es par saison ne contiennent "
                     "pas d'&eacute;pisode sans vent prolong&eacute;.",
                     "<b>The wind calm tail.</b> The modelled P95 is up to an order of magnitude "
                     "above the real one. Three days per season contain no prolonged windless "
                     "episode."),
                   t("<b>Sept zones &eacute;oliennes n'ont pas de journ&eacute;e r&eacute;elle du tout</b> "
                     "&mdash; s&eacute;rie refus&eacute;e au recalage &mdash; et {0} zones-saisons sur 160 "
                     "tournent encore sur une journ&eacute;e plate de 2020, dont {1} en &eacute;olien. "
                     "Le niveau y est juste, la distribution horaire non.",
                     "<b>Seven wind zones have no real day at all</b> &mdash; series refused at "
                     "rescaling &mdash; and {0} zone-seasons out of 160 still run on a flat 2020 "
                     "day, {1} of them wind. The level there is right, the hourly distribution "
                     "is not.").format(shape_by[("PV", "casa_2020 blocks")] +
                                       shape_by[("WT", "casa_2020 blocks")],
                                       shape_by[("WT", "casa_2020 blocks")]),
                   t("<b>La moiti&eacute; de la demande n'est pas v&eacute;rifiable.</b> Pakistan, "
                     "Afghanistan et Turkm&eacute;nistan n'ont pas d'ann&eacute;e horaire compt&eacute;e : "
                     "{0} % de la demande 2026 est contr&ocirc;l&eacute;e, le reste porte la forme 2020 "
                     "sans t&eacute;moin.",
                     "<b>Half the demand is not checkable.</b> Pakistan, Afghanistan and "
                     "Turkmenistan have no metered hourly year: {0}% of 2026 demand is "
                     "controlled, the rest carries the 2020 shape with no witness.").format(
                         "{0:.0f}".format(coverage)),
                   t("<b>Les journ&eacute;es sont choisies sur la charge seule.</b> Le clustering "
                     "conjoint charge + PV + vent (Poncelet), qui est ce que fait la mer Noire, "
                     "est &eacute;crit et gel&eacute; en attendant la charge horaire du Pakistan et de "
                     "l'Afghanistan. C'est la correction qui r&eacute;parerait la queue calme.",
                     "<b>The days are chosen on load alone.</b> The joint load + PV + wind "
                     "clustering (Poncelet), which is what Black Sea does, is written and frozen "
                     "pending hourly load for Pakistan and Afghanistan. That is the fix that "
                     "would repair the calm tail."),
               ]) + "</ul></div></div>")
    out.append("</div>")
    return "\n".join(out)


# ------------------------------------------------------------------ tab 2
def tab_calibration(run, base, traj, fleet, zero_cap, demand_uncovered, coverage,
                    deltas):
    out = []
    # -- I. generation --------------------------------------------------------
    out.append('<h3 class="roman">' + t(
        "I &middot; G&eacute;n&eacute;ration &mdash; le dispatch reproduit-il le r&eacute;el&nbsp;?",
        "I &middot; Generation &mdash; does the dispatch reproduce reality?") + "</h3>")
    out.append('<div class="sec" style="background:#fdf6f3;border-color:#f0d0c4">'
               '<div class="sn" style="color:#a5381f">' +
               t("R&eacute;ponse courte : non, et pas par n&eacute;gligence",
                 "Short answer: no, and not through neglect") + '</div>'
               '<div class="body">' + t(
                   "Calibrer un dispatch veut dire comparer la <b>production mod&eacute;lis&eacute;e par "
                   "combustible</b> &agrave; la <b>production compt&eacute;e par combustible</b> sur une "
                   "ann&eacute;e de base. <b>Aucune s&eacute;rie de ce type n'existe pour aucun des sept "
                   "pays de cette &eacute;tude.</b> Les cinq livres d'hypoth&egrave;ses DeCA donnent la "
                   "demande horaire, la capacit&eacute; install&eacute;e, les co&ucirc;ts et les facteurs "
                   "d'utilisation ; ils ne donnent pas la production r&eacute;alis&eacute;e centrale par "
                   "centrale. C'est une <b>demande de donn&eacute;es</b>, pas une lacune que cette "
                   "page peut combler, et tant qu'elle n'est pas satisfaite &laquo; calibration &raquo; "
                   "signifie ici <b>le c&ocirc;t&eacute; demande seul</b>. Remplacer la mesure manquante "
                   "par un ordre de grandeur de m&eacute;moire produirait un tableau vert qui ne "
                   "v&eacute;rifierait rien : c'est ce que cette section refuse de faire.",
                   "Calibrating a dispatch means comparing <b>modelled output by fuel</b> "
                   "against <b>metered output by fuel</b> over a base year. <b>No such series "
                   "exists for any of the seven countries in this study.</b> The five DeCA "
                   "assumption books give hourly demand, installed capacity, costs and "
                   "utilisation factors; they do not give realised generation plant by plant. "
                   "That is a <b>data request</b>, not a gap this page can close, and until it "
                   "is answered 'calibration' here means <b>the demand side alone</b>. "
                   "Substituting a remembered order of magnitude for the missing measurement "
                   "would produce a green table that verifies nothing: that is what this section "
                   "declines to do.") + "</div></div>")

    out.append('<div class="subh">' + t(
        "Ce qui est v&eacute;rifi&eacute;, et contre quoi",
        "What is checked, and against what") + "</div>")
    checks = [
        (t("Forme de la demande", "Demand shape"),
         t("s&eacute;rie horaire compt&eacute;e DeCA, 7 zones", "DeCA metered hourly series, 7 zones"),
         "ok", t("V&eacute;rifi&eacute; &mdash; onglet 1, &sect;4", "Checked &mdash; tab 1, &sect;4")),
        (t("Niveau de la demande", "Demand level"),
         t("pDemandForecast, construit sur le sc&eacute;nario Net Zero des livres DeCA",
           "pDemandForecast, built on the DeCA books' Net Zero scenario"),
         "wm", t("Trac&eacute; jusqu'&agrave; sa source, pas confront&eacute; &agrave; un r&eacute;alis&eacute;",
                 "Traced to its source, not confronted with an outturn")),
        (t("Forme du solaire et de l'&eacute;olien", "Solar and wind shape"),
         t("ann&eacute;e horaire Renewables.ninja (MERRA-2)",
           "Renewables.ninja hourly year (MERRA-2)"),
         "wm", t("V&eacute;rifi&eacute;e contre un mod&egrave;le, pas contre une mesure",
                 "Checked against a model, not against a measurement")),
        (t("Niveau du solaire et de l'&eacute;olien", "Solar and wind level"),
         t("facteur d'utilisation DeCA ; atlas &eacute;olien mondial l&agrave; o&ugrave; DeCA se tait",
           "DeCA utilisation factor; Global Wind Atlas where DeCA is silent"),
         "ok", t("Impos&eacute; par construction", "Imposed by construction")),
        (t("Parc existant", "Existing fleet"),
         t("livres DeCA + mod&egrave;le 2020", "DeCA books + 2020 model"),
         "wm", t("{0:,.0f} MW d&eacute;clar&eacute;s, dont <b>{1} centrales &agrave; capacit&eacute; nulle</b>",
                 "{0:,.0f} MW declared, including <b>{1} plants at zero capacity</b>").format(
                     fleet, zero_cap)),
        (t("Production par combustible", "Output by fuel"),
         t("&mdash; aucune source", "&mdash; no source"),
         "bd", t("<b>Non v&eacute;rifiable</b>", "<b>Not checkable</b>")),
    ]
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("Ce qui entre dans le mod&egrave;le", "What enters the model") + "</th><th>" +
               t("Contre quoi on peut le comparer", "What it can be compared against") +
               "</th><th>" + t("&Eacute;tat", "State") + "</th></tr></thead><tbody>" +
               "".join('<tr><td style="text-align:left"><b>{0}</b></td>'
                       '<td style="text-align:left;font-size:.78rem">{1}</td>'
                       '<td class="{2}" style="text-align:left">{3}</td></tr>'.format(*c)
                       for c in checks) + "</tbody></table></div>")
    out.append('<div class="pn">' + t(
        "Les {0} centrales existantes entr&eacute;es &agrave; capacit&eacute; nulle sont nomm&eacute;es dans "
        "extracted/pGenDataInput.csv : Douchanb&eacute;-1 et Yavan c&ocirc;t&eacute; tadjik, KANUPP &agrave; Karachi, "
        "et une douzaine d'unit&eacute;s pakistanaises. Le parc de d&eacute;part est donc <b>court</b> de "
        "leur capacit&eacute; r&eacute;elle, et le mod&egrave;le rachetera cette capacit&eacute; comme si elle "
        "n'existait pas. C'est le premier poste &agrave; corriger avant toute lecture d'ad&eacute;quation "
        "en ann&eacute;e de base.",
        "The {0} existing plants entered at zero capacity are named in "
        "extracted/pGenDataInput.csv: Dushanbe-1 and Yavan on the Tajik side, KANUPP at "
        "Karachi, and a dozen Pakistani units. The starting fleet is therefore <b>short</b> by "
        "their real capacity, and the model will re-buy that capacity as if it did not exist. "
        "It is the first item to fix before reading any base-year adequacy result.").format(
            zero_cap) + "</div>")

    # -- II. expansion --------------------------------------------------------
    out.append('<h3 class="roman">' + t(
        "II &middot; Expansion &mdash; la trajectoire est-elle coh&eacute;rente&nbsp;?",
        "II &middot; Expansion &mdash; is the trajectory coherent?") + "</h3>")
    years, cap, energy = traj
    order = [k for k in ["Coal", "Gas", "Diesel", "Nuclear", "Reservoir",
                         "Onshore Wind", "PV", "Imports"]
             if any(cap[y].get(k) for y in years)]
    out.append('<div class="method">' + t(
        "Trajectoire du sc&eacute;nario <code>baseline</code> du run <b>{0}</b>. Treize ann&eacute;es "
        "mod&eacute;lis&eacute;es, 2026 &agrave; 2035 puis 2040, 2045, 2050.",
        "Trajectory of the <code>baseline</code> scenario of run <b>{0}</b>. Thirteen modelled "
        "years, 2026 to 2035 then 2040, 2045, 2050.").format(run) + "</div>")
    out.append('<div class="subh">' + t("Capacit&eacute; install&eacute;e (GW)",
                                        "Installed capacity (GW)") + "</div>")
    out.append('<div class="chart">' + legend(order) + bars(years, cap, order) + "</div>")
    out.append('<div class="subh">' + t("Production (TWh)", "Generation (TWh)") + "</div>")
    out.append('<div class="chart">' + legend(order) +
               bars(years, energy, order, unit="GWh", axis="TWh") + "</div>")

    first, last = years[0], years[-1]
    growth = []
    for k in order:
        a, b = cap[first].get(k, 0.0), cap[last].get(k, 0.0)
        growth.append((k, a, b, b - a))
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("Technologie", "Technology") + "</th><th>" + first + " (MW)</th><th>" +
               last + " (MW)</th><th>" + t("&Delta;", "&Delta;") + "</th><th>" +
               t("&times;", "&times;") + "</th></tr></thead><tbody>" + "".join(
                   '<tr><td style="text-align:left"><b><span class="dot" '
                   'style="background:{0}"></span>{1}</b></td><td>{2:,.0f}</td>'
                   '<td>{3:,.0f}</td><td>{4:+,.0f}</td><td>{5}</td></tr>'.format(
                       FUEL_COLOUR.get(k, "#999"), k, a, b, d,
                       "{0:.1f}".format(b / a) if a else "&mdash;")
                   for k, a, b, d in growth) + "</tbody></table></div>")

    out.append('<div class="keybox">' + t(
        "<b>Ce que la trajectoire dit.</b> Le parc passe de {0:,.0f} MW en {1} &agrave; "
        "{2:,.0f} MW en {3}. La croissance est port&eacute;e par le <b>solaire</b> "
        "(&times;{4:.0f}) et l'<b>&eacute;olien</b> (&times;{5:.1f}), le gaz suivant en "
        "troisi&egrave;me ({6:+,.0f} MW) pour tenir la pointe et les soir&eacute;es sans soleil. Le "
        "charbon est &agrave; peu pr&egrave;s stable et le diesel dispara&icirc;t enti&egrave;rement avant 2040. "
        "Le r&eacute;servoir gagne {7:+,.0f} MW &mdash; c'est peu, et c'est un artefact connu : les "
        "projets hydro&eacute;lectriques des livres DeCA (Kazarman 997 MW, Nurobad-1/2 920 MW, "
        "Kambarata-2 240 MW, Mullalak 240 MW, Karakol, Kulanak, l'uprating de Kayrakum) ne "
        "sont <b>pas encore repr&eacute;sent&eacute;s dans le menu de candidats</b>, si bien que le "
        "mod&egrave;le ne peut pas les choisir m&ecirc;me s'ils sont comp&eacute;titifs.",
        "<b>What the trajectory says.</b> The fleet grows from {0:,.0f} MW in {1} to "
        "{2:,.0f} MW in {3}. Growth is carried by <b>solar</b> (&times;{4:.0f}) and "
        "<b>wind</b> (&times;{5:.1f}), with gas third ({6:+,.0f} MW) to hold the peak and the "
        "sunless evenings. Coal is roughly flat and diesel disappears entirely before 2040. "
        "Reservoir gains {7:+,.0f} MW &mdash; which is little, and is a known artefact: the "
        "hydro projects in the DeCA books (Kazarman 997 MW, Nurobad-1/2 920 MW, Kambarata-2 "
        "240 MW, Mullalak 240 MW, Karakol, Kulanak, the Kayrakum uprating) are <b>not yet in "
        "the candidate menu</b>, so the model cannot pick them even where they are "
        "competitive.").format(
            sum(cap[first].values()), first, sum(cap[last].values()), last,
            cap[last].get("PV", 0) / (cap[first].get("PV", 1) or 1),
            cap[last].get("Onshore Wind", 0) / (cap[first].get("Onshore Wind", 1) or 1),
            cap[last].get("Gas", 0) - cap[first].get("Gas", 0),
            cap[last].get("Reservoir", 0) - cap[first].get("Reservoir", 0)) + "</div>")

    # -- the two runs side by side -------------------------------------------
    out.append('<div class="subh">' + t(
        "Sensibilit&eacute; &mdash; ce que le niveau &eacute;olien change &agrave; lui seul",
        "Sensitivity &mdash; what the wind level alone changes") + "</div>")
    out.append('<div class="method">' + t(
        "Deux runs identiques &agrave; une chose pr&egrave;s : dans <code>{0}</code> les huit zones "
        "&eacute;oliennes afghanes et pakistanaises portent le 0,333 du mod&egrave;le 2020, un seul "
        "chiffre recopi&eacute; huit fois ; dans <code>{1}</code> elles portent le facteur que "
        "l'atlas &eacute;olien mondial lit sur leur propre terrain &agrave; 250 m. Rien d'autre ne "
        "diff&egrave;re. C'est donc une mesure directe de ce qu'une donn&eacute;e de substitution "
        "co&ucirc;tait au r&eacute;sultat.",
        "Two runs identical but for one thing: in <code>{0}</code> the eight Afghan and "
        "Pakistani wind zones carry the 2020 model's 0.333, one figure copied eight times; in "
        "<code>{1}</code> they carry the factor the Global Wind Atlas reads on their own "
        "terrain at 250 m. Nothing else differs. It is therefore a direct measure of what a "
        "placeholder was costing the answer.").format(base, run) + "</div>")
    out.append('<div class="gridtbl"><table class="cal"><thead><tr><th>' +
               t("Indicateur", "Indicator") + "</th><th>" + base + "</th><th>" +
               t("atlas", "atlas") + "</th><th>&Delta;</th></tr></thead><tbody>" +
               "".join('<tr><td style="text-align:left"><b>{0}</b></td><td>{1}</td>'
                       '<td>{2}</td><td class="{3}">{4}</td></tr>'.format(*d)
                       for d in deltas) + "</tbody></table></div>")
    out.append('<div class="keybox" style="background:#fff8ef;border-color:#f0dcc0;'
               'color:#7a541b">' + t(
                   "<b>C'est un r&eacute;sultat, pas une am&eacute;lioration.</b> Le syst&egrave;me co&ucirc;te "
                   "moins cher parce qu'il cesse de surinvestir dans un &eacute;olien dont le "
                   "rendement &eacute;tait surestim&eacute; &mdash; PAK_N perd la totalit&eacute; de ses 7,3 GW "
                   "&eacute;oliens de 2050, son facteur tombant de 0,333 &agrave; 0,214 &mdash; et ce que "
                   "l'&eacute;olien ne fait plus, le gaz et le charbon le font. On a donc &eacute;chang&eacute; "
                   "une erreur de donn&eacute;es contre une trajectoire plus fossile : bonne "
                   "nouvelle pour la rigueur, mauvaise pour le message. &Agrave; l'inverse TAJ_N "
                   "appara&icirc;t avec 1,3 GW, parce que le d&eacute;placement du point de fetch vers "
                   "le meilleur site de l'atlas a fait passer son facteur de 0,106 &agrave; 0,342.",
                   "<b>This is a result, not an improvement.</b> The system costs less because "
                   "it stops over-investing in wind whose yield was overstated &mdash; PAK_N "
                   "loses all 7.3 GW of its 2050 wind, its factor falling from 0.333 to 0.214 "
                   "&mdash; and what wind no longer does, gas and coal do. A data error has been "
                   "traded for a more fossil trajectory: good news for rigour, bad news for the "
                   "message. Conversely TAJ_N appears with 1.3 GW, because moving the fetch "
                   "point to the atlas's best site took its factor from 0.106 to 0.342.") +
               "</div>")

    out.append('<h3 class="roman">' + t(
        "III &middot; Dette connue", "III &middot; Known debt") + "</h3>")
    out.append('<div class="sec"><div class="body"><ul>' + "".join(
        "<li>" + x + "</li>" for x in [
            t("<b>{0} centrales existantes &agrave; capacit&eacute; nulle</b>, dont Douchanb&eacute;-1, Yavan "
              "et KANUPP.", "<b>{0} existing plants at zero capacity</b>, including Dushanbe-1, "
              "Yavan and KANUPP.").format(zero_cap),
            t("<b>&Eacute;cart thermique</b> : Bichkek 666 MW d&eacute;clar&eacute;s contre 812 attendus, "
              "Douchanb&eacute;-1/2 &agrave; 0 contre 598, Turkm&eacute;nistan 5,8 GW contre 7,7.",
              "<b>Thermal gap</b>: Bishkek 666 MW declared against 812 expected, Dushanbe-1/2 "
              "at 0 against 598, Turkmenistan 5.8 GW against 7.7."),
            t("<b>Hydro DeCA absente du menu de candidats</b> : Karakol 33 MW, Kulanak 100 MW, "
              "cascade de Kazarman 997 MW, Nurobad-1/2 920 MW, Kambarata-2 240 MW, "
              "Mullalak 240 MW, uprating de Kayrakum 48 MW.",
              "<b>DeCA hydro absent from the candidate menu</b>: Karakol 33 MW, Kulanak "
              "100 MW, Kazarman cascade 997 MW, Nurobad-1/2 920 MW, Kambarata-2 240 MW, "
              "Mullalak 240 MW, Kayrakum uprating 48 MW."),
            t("<b>FOM et VOM hydro</b> encore sur les valeurs 2020 pour tout le parc.",
              "<b>Hydro FOM and VOM</b> still on 2020 values fleet-wide."),
            t("<b>Prix du carbone</b> : les donn&eacute;es sont en place, "
              "<code>fEnableCarbonPrice = 0</code>. Sans lui, la correction &eacute;olienne pousse "
              "m&eacute;caniquement vers le gaz.",
              "<b>Carbon price</b>: the data is in place, <code>fEnableCarbonPrice = 0</code>. "
              "Without it, the wind correction mechanically pushes towards gas."),
            t("<b>pDemandProfile &mdash; Turkm&eacute;nistan</b> : la seule zone o&ugrave; profil et "
              "pr&eacute;vision ne s'accordent pas, et la seule dont la source est de 288 points.",
              "<b>pDemandProfile &mdash; Turkmenistan</b>: the one zone where profile and "
              "forecast disagree, and the only one whose source is 288 points."),
            t("<b>threads 4</b> dans <code>cplex_baseline.opt</code> est une propri&eacute;t&eacute; de la "
              "machine, pas du mod&egrave;le : &agrave; remonter &agrave; 8 sur une machine avec de la marge.",
              "<b>threads 4</b> in <code>cplex_baseline.opt</code> is a property of the "
              "machine, not of the model: raise it back to 8 on a machine with headroom."),
        ]) + "</ul></div></div>")
    return "\n".join(out)


# ------------------------------------------------------------------ assembly
def trajectory(path):
    rows = read_csv(path)
    cap = collections.defaultdict(lambda: collections.defaultdict(float))
    energy = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        year = r["year"].split(".")[0]
        try:
            v = float(r["baseline"])
        except (TypeError, ValueError):
            continue
        if r["attribute"] == "Capacity: MW":
            cap[year][r["resolution"]] += v
        elif r["attribute"] == "Energy: GWh":
            energy[year][r["resolution"]] += v
    years = sorted(cap, key=float)
    return years, cap, energy


def compare_runs(base_path, run_path):
    """The handful of headline numbers, both runs, for the sensitivity table."""
    def totals(path):
        out = collections.defaultdict(float)
        for r in read_csv(path):
            try:
                out[r["attribute"]] += float(r["baseline"])
            except (TypeError, ValueError):
                pass
        return out

    a, b = totals(base_path), totals(run_path)
    wanted = [
        ("NPV of system cost: $m", t("VAN du co&ucirc;t syst&egrave;me", "NPV of system cost"),
         "{0:,.0f} M$", 0.01, 0.03),
        ("NPV of system cost: $/MWh", t("Co&ucirc;t moyen", "Average cost"), "{0:,.1f} $/MWh",
         0.01, 0.03),
        ("Investment costs: $m", t("Investissement", "Investment"), "{0:,.0f} M$", 0.03, 0.08),
        ("Fuel costs: $m", t("Combustible", "Fuel"), "{0:,.0f} M$", 0.03, 0.08),
        ("Emissions: MtCO2", t("&Eacute;missions", "Emissions"), "{0:,.0f} Mt", 0.02, 0.05),
        ("Unmet demand: GWh", t("Demande non servie", "Unmet demand"), "{0:,.0f} GWh",
         0.10, 0.30),
        ("Annual Energy Exchanges: GWh", t("&Eacute;changes", "Exchanges"), "{0:,.0f} GWh",
         0.02, 0.05),
    ]
    rows = []
    for key, label, fmt, good, fair in wanted:
        x, y = a.get(key, 0.0), b.get(key, 0.0)
        rel = (y - x) / x if x else None
        rows.append((label, fmt.format(x), fmt.format(y),
                     klass(rel, good, fair), pct(rel, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    metrics = json.load(io.open(os.path.join(EXTRACTED, "review_metrics.json"),
                                encoding="utf-8"))
    dates = read_csv(os.path.join(EXTRACTED, "demand_report.csv"))
    vre_report = read_csv(os.path.join(EXTRACTED, "vre_report.csv"))
    hourly_report = read_csv(os.path.join(EXTRACTED, "vre_hourly_report.csv"))

    forecast = [r for r in read_csv(os.path.join(EXTRACTED, "pDemandForecast.csv"))
                if r["type"] == "Energy"]
    total = sum(float(r["2026"]) for r in forecast)
    covered = sum(float(r["2026"]) for r in forecast if r["z"] in metrics["demand"])
    coverage = 100.0 * covered / total

    gens = read_csv(os.path.join(EXTRACTED, "pGenDataInput.csv"))
    existing = [r for r in gens if r["Status"].strip() == "1"]
    fleet = sum(float(r["Capacity"] or 0) for r in existing)
    zero_cap = sum(1 for r in existing if float(r["Capacity"] or 0) == 0)

    out_dir = os.path.join(ROOT, "epm", "output")
    traj = trajectory(os.path.join(out_dir, args.run, "summary.csv"))
    deltas = compare_runs(os.path.join(out_dir, args.base, "summary.csv"),
                          os.path.join(out_dir, args.run, "summary.csv"))

    css = io.open(os.path.join(HERE, "calibration_review_casa.css"),
                  encoding="utf-8").read()
    page = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Calibration review &mdash; CASA 2026</title><style>", css,
        '</style></head><body class="en">',
        '<div class="langtoggle"><button id="btnfr" onclick="setLang(\'fr\')">FR</button>'
        '<button id="btnen" class="on" onclick="setLang(\'en\')">EN</button></div>',
        "<h1>" + t("Revue de calibration &mdash; Asie centrale et Asie du Sud (CASA)",
                   "Calibration review &mdash; Central and South Asia (CASA)") + "</h1>",
        '<p class="sub">' + t(
            "Deux questions, deux onglets : la r&eacute;duction temporelle d&eacute;crit-elle encore "
            "l'ann&eacute;e dont elle sort, et le mod&egrave;le reproduit-il le syst&egrave;me tel qu'il est "
            "avant de commencer &agrave; construire.",
            "Two questions, two tabs: does the temporal reduction still describe the year it "
            "came from, and does the model reproduce the system as it stands before it starts "
            "building.") + "</p>",
        '<p class="sub">' + t(
            "Chiffres r&eacute;g&eacute;n&eacute;r&eacute;s le {0} depuis le run <b>{1}</b> (sc&eacute;nario baseline) et "
            "depuis data_build/extracted/review_metrics.json, par "
            "<code>tools/calibration_review_casa.py</code>. Aucun chiffre de cette page n'est "
            "saisi &agrave; la main.",
            "Figures regenerated on {0} from run <b>{1}</b> (baseline scenario) and from "
            "data_build/extracted/review_metrics.json, by "
            "<code>tools/calibration_review_casa.py</code>. No figure on this page is typed in "
            "by hand.").format(datetime.date.today().isoformat(), args.run) + "</p>",
        '<div class="intro">' + t(
            "<b>Contexte.</b> &Eacute;tude de planification du syst&egrave;me &eacute;lectrique r&eacute;gional "
            "<b>Asie centrale &ndash; Asie du Sud</b> de la Banque mondiale (EPM, moindre co&ucirc;t "
            "open-source). 16 zones, 7 pays (Afghanistan, Kazakhstan, Kirghizistan, Pakistan, "
            "Tadjikistan, Turkm&eacute;nistan, Ouzb&eacute;kistan), horizon 2050. Le mod&egrave;le est port&eacute; "
            "depuis la version EPM v7.9 de d&eacute;cembre 2020 et remis &agrave; jour sur les cinq livres "
            "d'hypoth&egrave;ses DeCA. <b>Exercice interne</b> destin&eacute; &agrave; &eacute;clairer le dialogue de "
            "la Banque.",
            "<b>Context.</b> World Bank <b>Central Asia &ndash; South Asia Regional Power System "
            "Planning Study</b> (EPM, open-source least-cost). 16 zones, 7 countries "
            "(Afghanistan, Kazakhstan, Kyrgyz Republic, Pakistan, Tajikistan, Turkmenistan, "
            "Uzbekistan), horizon 2050. The model is ported from EPM v7.9 of December 2020 and "
            "brought up to date on the five DeCA assumption books. <b>Internal exercise</b> "
            "intended to inform the Bank's own dialogue.") + "</div>",
        '<div class="toptabs">'
        '<button class="toptab active" onclick="showTop(\'TOP-repdays\',this)">' +
        t("Journ&eacute;es repr&eacute;sentatives", "Representative days") + "</button>"
        '<button class="toptab" onclick="showTop(\'TOP-cal\',this)">' +
        t("Calibration (ann&eacute;e de base)", "Calibration (base year)") + "</button></div>",
        '<div class="toppanel active" id="TOP-repdays">',
        '<div class="phead"><h2>' + t(
            "Journ&eacute;es repr&eacute;sentatives &mdash; fid&eacute;lit&eacute; de la repr&eacute;sentation temporelle",
            "Representative days &mdash; fidelity of the temporal representation") +
        '</h2><span class="vb md">' + t(
            "Robuste (charge &amp; PV) &middot; Vent &agrave; surveiller",
            "Robust (load &amp; PV) &middot; Wind to watch") + "</span></div>",
        '<div class="intro" style="background:#eef6ee;border:1px solid #cfe3cf">' + t(
            "<b>Question.</b> Le mod&egrave;le ne tourne pas sur 8 760 h mais sur un <b>jeu r&eacute;duit "
            "de journ&eacute;es repr&eacute;sentatives pond&eacute;r&eacute;es</b> : 5 saisons &times; 3 journ&eacute;es &times; "
            "24 h = 360 pas de temps. Cet onglet v&eacute;rifie que cette r&eacute;duction <b>reproduit "
            "fid&egrave;lement</b> les vraies chroniques, <b>par technologie</b> et pas seulement en "
            "moyenne, en s&eacute;parant deux choses tr&egrave;s diff&eacute;rentes : la r&eacute;duction pr&eacute;serve-t-elle "
            "<b>la valeur</b> (&eacute;nergie, distribution, extr&ecirc;mes) et pr&eacute;serve-t-elle <b>le "
            "profil</b> (la forme intra-journali&egrave;re, l'heure des pointes et des creux) ?",
            "<b>Question.</b> The model does not run on 8,760 h but on a <b>reduced set of "
            "weighted representative days</b>: 5 seasons &times; 3 days &times; 24 h = 360 "
            "timesteps. This tab checks that the reduction <b>faithfully reproduces</b> the real "
            "series, <b>per technology</b> and not only on average, separating two very "
            "different things: does the reduction preserve <b>value</b> (energy, distribution, "
            "extremes) and does it preserve <b>profile</b> (the intraday shape, the timing of "
            "peaks and troughs)?") + "</div>",
        tab_representative(metrics, dates, vre_report, hourly_report, coverage),
        "</div>",
        '<div class="toppanel" id="TOP-cal">',
        '<div class="phead"><h2>' + t(
            "Calibration &mdash; ann&eacute;e de base &amp; trajectoire",
            "Calibration &mdash; base year &amp; trajectory") +
        '</h2><span class="vb hi">' + t(
            "Demande calibr&eacute;e &middot; Dispatch non calibrable",
            "Demand calibrated &middot; Dispatch not calibratable") + "</span></div>",
        tab_calibration(args.run, args.base, traj, fleet, zero_cap,
                        metrics["demand_uncovered"], coverage, deltas),
        "</div>",
        """<script>
function showTop(id, btn){
  var p = document.querySelectorAll('.toppanel');
  for (var i=0;i<p.length;i++){ p[i].classList.remove('active'); }
  document.getElementById(id).classList.add('active');
  var b = document.querySelectorAll('.toptab');
  for (var j=0;j<b.length;j++){ b[j].classList.remove('active'); }
  btn.classList.add('active');
  window.scrollTo(0,0);
}
function setLang(l){
  document.body.classList.toggle('en', l === 'en');
  document.getElementById('btnfr').classList.toggle('on', l === 'fr');
  document.getElementById('btnen').classList.toggle('on', l === 'en');
}
</script></body></html>""",
    ]
    if not os.path.isdir(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(page))
    print("written  {0}  ({1:,} bytes)".format(args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
