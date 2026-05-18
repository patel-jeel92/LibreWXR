# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""OPERA pan-European CIRRUS composite radar network.

Sourced verbatim from the official EUMETNET OPERA radar database
(``status="operational"`` only):
    https://www.eumetnet.eu/wp-content/themes/aeron-child/observations-programme/current-activities/opera/database/OPERA_Database/

Used for station-circle coverage masks to prevent ECMWF fallback
suppression across the entire LAEA grid bbox.  Regenerate this list by
downloading ``Data/OPERA_RADARS_DB_<date>.json`` from the OPERA Database
page and filtering on ``status="1"``.

Range override: European C-band radars in the OPERA network detect
precipitation out to ~300 km — using the 240 km default leaves gaps
around peripheral stations (Iceland, Ireland) where OPERA still has
data but the coverage mask says "not covered," causing visible
ECMWF/OPERA overlap seams.
"""
from __future__ import annotations


STATIONS: list[tuple[float, float]] = [
    # Belgium (BE) - 3 radars
    (51.0702, 5.4054),    # behel Helchteren
    (51.1919, 3.0641),    # bejab Jabbeke
    (49.9135, 5.5044),    # bewid Wideumont
    # Croatia (HR) - 6 radars
    (45.8835, 17.2005),   # hrbil Bilogora
    (44.0455, 15.3765),   # hrdeb Debeljak
    (45.0205, 14.1223),   # hrgol Goli
    (45.1592, 18.7033),   # hrgra Gradište
    (45.9078, 15.9683),   # hrpun Puntijarka
    (42.8944, 17.4783),   # hrulj Uljenje
    # Czechia (CZ) - 2 radars
    (49.6583, 13.8178),   # czbrd Brdy-Praha
    (49.5011, 16.7885),   # czska Skalky
    # Denmark (DK) - 4 radars
    (55.1127, 14.8875),   # dkbor Bornholm
    (55.1731, 8.5520),    # dkrom Römö
    (57.4893, 10.1365),   # dksin Sindal
    (55.3262, 12.4493),   # dkste Stevns
    # Estonia (EE) - 2 radars
    (59.3971, 24.6021),   # eehar Harku
    (58.4823, 25.5187),   # eesur Sürgavere
    # Finland (FI) - 12 radars
    (60.9039, 27.1081),   # fianj Anjalankoski
    (61.8108, 22.5020),   # fikan Kankaanpää
    (68.4344, 27.4440),   # fikau Kaunispää
    (61.9069, 29.7977),   # fikes Kesälahti
    (60.1285, 21.6434),   # fikor Korpo
    (62.8626, 27.3815),   # fikuo Kuopio
    (67.1391, 26.8969),   # filuo Luosto
    (63.8378, 29.4489),   # finur Nurmes
    (62.3045, 25.4401),   # fipet Petäjävesi
    (64.7749, 26.3189),   # fiuta Utajärvi
    (60.5562, 24.4956),   # fivih Vihti
    (63.1048, 23.8209),   # fivim Vimpeli
    # France (FR) - 25 radars
    (50.1360, 1.8347),    # frabb Abbeville
    (41.9531, 8.7005),    # fraja Ajaccio
    (42.1298, 9.4964),    # frale Aléria
    (50.1283, 3.8118),    # frave Avesnes
    (47.3552, 4.7759),    # frbla Blaisy-Haut
    (44.3230, 4.7621),    # frbol Bollène
    (44.8315, -0.6919),   # frbor Bordeaux
    (47.0586, 2.3595),    # frbou Bourges
    (48.9272, -0.1496),   # frcae Falaise
    (43.2166, 6.3729),    # frcol Collobrières
    (45.1044, 1.3697),    # frgre Grèzes
    (45.2892, 3.7095),    # frlep Sembadel
    (44.0128, 6.5292),    # frmau Mont Maurel
    (43.9905, 2.6096),    # frmcl Montclar
    (43.6245, -0.6094),   # frmom Momuy
    (47.3686, 7.0190),    # frmtc Montancy
    (48.7158, 6.5816),    # frnan Nancy
    (43.8061, 4.5027),    # frnim Nîmes
    (46.0678, 4.4453),    # frniz Saint Nizier
    (42.9184, 2.8650),    # fropo Opoul
    (48.4609, -4.4298),   # frpla Plabennec
    (43.5743, 1.3763),    # frtou Toulouse
    (48.7746, 2.0083),    # frtra Trappes
    (47.3374, -1.6563),   # frtre Treillères
    (48.4621, 4.3093),    # frtro Arcis-sur-Aube
    # Germany (DE) - 17 radars
    (53.5640, 6.7482),    # deasb Isle of Borkum
    (54.0043, 10.0468),   # deboo Boostedt
    (51.1246, 13.7686),   # dedrs Dresden
    (49.5407, 12.4028),   # deeis Eisberg
    (51.4055, 6.9669),    # deess Essen
    (47.8736, 8.0039),    # defbg Feldberg
    (51.3112, 8.8020),    # defld Flechtdorf
    (52.4600, 9.6945),    # dehnr Hannover
    (48.1747, 12.1017),   # deisn Isen/München
    (48.0421, 10.2192),   # demem Memmingen
    (50.5001, 11.1351),   # deneu Neuhaus
    (50.1097, 6.5483),    # denhb Neuheilenbach
    (49.9847, 8.7129),    # deoft Offenthal
    (52.6486, 13.8580),   # depro Protzel/Berlin
    (54.1757, 12.0580),   # deros Rostock
    (48.5853, 9.7828),    # detur Tuerkheim
    (52.1601, 11.1761),   # deumd Ummendorf
    # Greece (GR) - 5 radars
    (38.9166, 20.7528),   # grakt Aktio
    (37.9261, 21.2894),   # grand Andravida
    (39.6446, 22.4603),   # grlar Larisa
    (40.5282, 22.9757),   # grthe Thessaloniki
    (37.9461, 23.8138),   # grymi Ymittos
    # Hungary (HU) - 5 radars
    (47.4294, 19.1817),   # hubud Budapest
    (46.1775, 18.3372),   # huhar Harmashegy
    (47.9622, 21.8867),   # hunap Napkor
    (46.6604, 17.0624),   # hupog Poganyvar
    (46.6397, 20.4325),   # husze Szentes
    # Iceland (IS) - 3 radars
    (65.2658, -14.0618),  # isbjo Bjolfur
    (64.0257, -22.6353),  # iskef Keflavik
    (66.0557, -20.2680),  # isska Skagi
    # Ireland (IE) - 2 radars
    (53.4299, -6.2443),   # iedub Dublin
    (52.6928, -8.9201),   # iesha Shannon
    # Latvia (LV) - 1 radars
    (56.9143, 23.9897),   # lvrix Riga
    # Lithuania (LT) - 2 radars
    (55.6090, 22.2395),   # ltlau Laukuva
    (54.6262, 25.1067),   # ltvil Vilnius
    # Malta (MT) - 1 radars
    (35.8529, 14.4748),   # mtgud Gudja
    # Netherlands (NL) - 2 radars
    (52.9528, 4.7906),    # nldhl Den Helder
    (51.8371, 5.1380),    # nlhrw Herwijnen
    # Norway (NO) - 12 radars
    (69.2414, 16.0030),   # noand Andoya
    (70.5107, 29.0184),   # nober Berlevaag
    (59.8537, 5.0896),    # nobml Boemlo
    (70.6052, 22.4430),   # nohas Hasvik
    (61.2318, 10.5273),   # nohfj Hafjell
    (58.3601, 7.1648),    # nohgb Haegebostad
    (59.6272, 10.5645),   # nohur Hurum
    (63.6905, 10.2039),   # norsa Rissa
    (69.2186, 23.4398),   # norsg Rassegalvarri
    (67.5304, 12.0989),   # norst Rost
    (65.2199, 11.9925),   # nosmn Soemna
    (62.1871, 5.1275),    # nosta Stad
    # Poland (PL) - 10 radars
    (50.3942, 20.0832),   # plbrz Brzuchania
    (54.5009, 18.2718),   # plgdy Gdynia-Szemud
    (50.4639, 18.1532),   # plgsa Góra Św. Anny
    (52.4053, 20.9611),   # plleg Legionowo
    (50.8925, 16.0395),   # plpas Pastewnik
    (52.4133, 16.7970),   # plpoz Poznań
    (50.1513, 18.7251),   # plram Ramża
    (50.1141, 22.0370),   # plrze Rzeszów
    (53.7958, 15.8368),   # plswi Świdwin
    (53.8565, 21.4121),   # pluzr Uzranki
    # Portugal (PT) - 6 radars
    (37.3041, -7.9530),   # ptfar Loulé/Cavalos do Caldeirão
    (39.4634, -31.2200),  # ptflr Flores/Morro Alto
    (39.0714, -8.4001),   # ptlis Coruche/Cruz do Leão
    (40.8450, -8.2797),   # ptprt Arouca/Pico do Gralheiro
    (37.8191, -25.7516),  # ptsmg São Miguel/Pico Santos de Cima
    (38.7302, -27.3208),  # pttrc Terceira/ Santa Barbara
    # Romania (RO) - 7 radars
    (47.0118, 27.5826),   # robar Barnova
    (46.3602, 24.2252),   # robob Bobohalma
    (44.5127, 26.0773),   # robuc Bucuresti
    (44.3103, 23.8674),   # rocra Craiova
    (44.2434, 28.2506),   # romed Medgidia
    (47.0922, 21.9429),   # roora Oradea
    (45.7717, 21.2577),   # rotim Timisoara
    # Serbia (RS) - 3 radars
    (45.1573, 19.8109),   # rsfrg Fruska gora
    (43.3913, 21.4436),   # rsjas Jastrebac
    (45.1876, 20.7707),   # rssam Samos
    # Slovakia (SK) - 4 radars
    (48.2556, 17.1524),   # skjav Maly Javornik
    (48.7827, 20.9873),   # skkoj Kojsovska hola
    (49.2717, 19.2494),   # skkub Kubinska hola
    (48.2404, 19.2573),   # sklaz Spani laz
    # Slovenia (SI) - 2 radars
    (46.0680, 15.2850),   # silis Lisca
    (46.0980, 14.2283),   # sipas Pasja ravan
    # Spain (ES) - 15 radars
    (36.6133, -4.6593),   # esahr Alhaurin el Grande (Malaga)
    (36.8324, -2.0821),   # esalm Nijar (Almeria)
    (43.1690, -8.5269),   # escor Cerceda ( La Coruna)
    (41.4081, 1.8848),    # esgld Gelida (Barcelona)
    (41.9956, -4.6028),   # eslid Autilla Pino (Palencia)
    (28.0186, -15.6144),  # eslpa Artenara (Gran Canaria)
    (38.2644, -1.1897),   # esmur Fortuna (Murcia)
    (39.3797, 2.7851),    # espma Llucmajor (Baleares)
    (43.4625, -6.3019),   # essan Aguión (Asturias)
    (37.6887, -6.3331),   # essev Castillo las Guardas (Sevilla)
    (39.4288, -6.2853),   # essft Sierra de Fuentes (Caceres)
    (43.4033, -2.8419),   # essse Baquio (Vizcaya)
    (40.1759, -3.7136),   # estjv Torrejon de Velasco (Madrid)
    (39.1761, -0.2521),   # esval Cullera (Valencia)
    (41.7339, -0.5459),   # eszar Perdiguera (Zaragoza)
    # Sweden (SE) - 12 radars
    (56.3675, 12.8517),   # seang Ängelholm
    (58.1059, 15.9365),   # seatv Åtvidaberg (Vilebo)
    (59.6110, 17.5833),   # sebaa Bålsta
    (57.3035, 18.4001),   # sehem Hemse (Ase)
    (61.5771, 16.7144),   # sehuv Hudiksvall
    (56.2955, 15.6102),   # sekaa Karlskrona
    (67.7088, 20.6178),   # sekrn Kiruna
    (60.7230, 14.8776),   # selek Leksand
    (65.4309, 21.8650),   # sella Luleå (Rosvik)
    (63.6395, 18.4019),   # seoer Örnsköldsvik
    (63.2951, 14.7591),   # seosd Östersund
    (58.2556, 12.8260),   # sevax Vara
    # Switzerland (CH) - 5 radars
    (47.2843, 8.5120),    # chalb Albis
    (46.4251, 6.0994),    # chdol La Dole
    (46.0408, 8.8332),    # chlem Monte Lema
    (46.3706, 7.4866),    # chppm Plaine Monte
    (46.8350, 9.7945),    # chwei Weissfluhgiptel
    # United Kingdom (UK) - 16 radars
    (54.5020, -6.3429),   # ukcas Castor Bay
    (51.6895, -0.5302),   # ukche Chenies
    (52.3986, -2.5954),   # ukcle Clee Hill
    (50.9639, -3.4521),   # ukcob Cobbacombe Cross
    (51.9798, -4.4446),   # ukcyg Crug y Gorllwyn
    (51.0307, -1.6534),   # ukdea Dean Hill
    (57.4304, -2.0367),   # ukdud Hill Of Dudwick
    (53.7548, -2.2892),   # ukham Hameldon Hill
    (56.0180, -4.2171),   # ukhhd Holehead
    (54.8034, -1.4746),   # ukhmy High Moorsley
    (53.3347, -0.5594),   # uking Ingham
    (49.1772, -2.2239),   # ukjer Jersey
    (58.2117, -6.1821),   # uklew Druim A'Starraig
    (56.2148, -3.3118),   # ukmun Munduff Hill
    (50.0034, -5.2230),   # ukpre Predannack
    (51.2948, 0.6043),    # ukthu Thurnham
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "OPERA": STATIONS,
}

RANGE_OVERRIDES: dict[str, float] = {
    "OPERA": 300.0,
}
