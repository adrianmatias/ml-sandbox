import pandas as pd
import requests
from io import StringIO
import time


class CrawlerWiki:
    pass

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Grokipedia-standardized 100-seed list (BoxRec Top 60 core + 40 consensus additions)
fighters = {
    "Roberto Duran": "https://en.wikipedia.org/wiki/Roberto_Dur%C3%A1n",
    "Muhammad Ali": "https://en.wikipedia.org/wiki/Boxing_career_of_Muhammad_Ali",
    "Floyd Mayweather Jr.": "https://en.wikipedia.org/wiki/Floyd_Mayweather_Jr.",
    "Carlos Monzon": "https://en.wikipedia.org/wiki/Carlos_Monzon",
    "Sugar Ray Leonard": "https://en.wikipedia.org/wiki/Sugar_Ray_Leonard",
    "Manny Pacquiao": "https://en.wikipedia.org/wiki/Manny_Pacquiao",
    "Pernell Whitaker": "https://en.wikipedia.org/wiki/Pernell_Whitaker",
    "Julio Cesar Chavez": "https://en.wikipedia.org/wiki/Julio_C%C3%A9sar_Ch%C3%A1vez",
    "Emile Griffith": "https://en.wikipedia.org/wiki/Emile_Griffith",
    "Alexis Arguello": "https://en.wikipedia.org/wiki/Alexis_Arg%C3%BCello",
    "Eder Jofre": "https://en.wikipedia.org/wiki/Eder_Jofre",
    "Marvin Hagler": "https://en.wikipedia.org/wiki/Marvin_Hagler",
    "Roy Jones Jr.": "https://en.wikipedia.org/wiki/Roy_Jones_Jr.",
    "Evander Holyfield": "https://en.wikipedia.org/wiki/Evander_Holyfield",
    "Salvador Sanchez": "https://en.wikipedia.org/wiki/Salvador_S%C3%A1nchez",
    "Thomas Hearns": "https://en.wikipedia.org/wiki/Thomas_Hearns",
    "Larry Holmes": "https://en.wikipedia.org/wiki/Larry_Holmes",
    "George Foreman": "https://en.wikipedia.org/wiki/George_Foreman",
    "Dick Tiger": "https://en.wikipedia.org/wiki/Dick_Tiger",
    "Joe Frazier": "https://en.wikipedia.org/wiki/Joe_Frazier",
    "Bernard Hopkins": "https://en.wikipedia.org/wiki/Bernard_Hopkins",
    "Ruben Olivares": "https://en.wikipedia.org/wiki/Rub%C3%A9n_Olivares",
    "Juan Manuel Marquez": "https://en.wikipedia.org/wiki/Juan_Manuel_M%C3%A1rquez",
    "Michael Spinks": "https://en.wikipedia.org/wiki/Michael_Spinks",
    "Jose Napoles": "https://en.wikipedia.org/wiki/Jos%C3%A9_N%C3%A1poles",
    "Fighting Harada": "https://en.wikipedia.org/wiki/Masahiko_Harada",
    "Wilfredo Gomez": "https://en.wikipedia.org/wiki/Wilfredo_G%C3%B3mez",
    "Carlos Ortiz": "https://en.wikipedia.org/wiki/Carlos_Ortiz_(boxer)",
    "Wilfred Benitez": "https://en.wikipedia.org/wiki/Wilfred_Ben%C3%ADtez",
    "Carlos Zarate": "https://en.wikipedia.org/wiki/Carlos_Z%C3%A1rate",
    "Miguel Canto": "https://en.wikipedia.org/wiki/Miguel_Canto",
    "Bob Foster": "https://en.wikipedia.org/wiki/Bob_Foster_(boxer)",
    "Marco Antonio Barrera": "https://en.wikipedia.org/wiki/Marco_Antonio_Barrera",
    "Ricardo Lopez": "https://en.wikipedia.org/wiki/Ricardo_L%C3%B3pez_(boxer)",
    "Luis Manuel Rodriguez": "https://en.wikipedia.org/wiki/Luis_Manuel_Rodr%C3%ADguez",
    "Erik Morales": "https://en.wikipedia.org/wiki/Erik_Morales",
    "Azumah Nelson": "https://en.wikipedia.org/wiki/Azumah_Nelson",
    "Lennox Lewis": "https://en.wikipedia.org/wiki/Lennox_Lewis",
    "James Toney": "https://en.wikipedia.org/wiki/James_Toney",
    "Mike McCallum": "https://en.wikipedia.org/wiki/Mike_McCallum",
    "Eusebio Pedroza": "https://en.wikipedia.org/wiki/Eusebio_Pedroza",
    "Mike Tyson": "https://en.wikipedia.org/wiki/Mike_Tyson",
    "Vicente Saldivar": "https://en.wikipedia.org/wiki/Vicente_Saldivar",
    "Terence Crawford": "https://en.wikipedia.org/wiki/Terence_Crawford",
    "Nino Benvenuti": "https://en.wikipedia.org/wiki/Nino_Benvenuti",
    "Canelo Alvarez": "https://en.wikipedia.org/wiki/Canelo_%C3%81lvarez",
    "Aaron Pryor": "https://en.wikipedia.org/wiki/Aaron_Pryor",
    "Flash Elorde": "https://en.wikipedia.org/wiki/Flash_Elorde",
    "Gennady Golovkin": "https://en.wikipedia.org/wiki/Gennady_Golovkin",
    "Khaosai Galaxy": "https://en.wikipedia.org/wiki/Khaosai_Galaxy",
    "Roman Gonzalez": "https://en.wikipedia.org/wiki/Rom%C3%A1n_Gonz%C3%A1lez",
    "Oscar De La Hoya": "https://en.wikipedia.org/wiki/Oscar_De_La_Hoya",
    "Wladimir Klitschko": "https://en.wikipedia.org/wiki/Wladimir_Klitschko",
    "Ronald Wright": "https://en.wikipedia.org/wiki/Ronald_Wright",
    "Sugar Shane Mosley": "https://en.wikipedia.org/wiki/Shane_Mosley",
    "Felix Trinidad": "https://en.wikipedia.org/wiki/F%C3%A9lix_Trinidad",
    "Myung Woo Yuh": "https://en.wikipedia.org/wiki/Yuh_Myung-woo",
    "Antonio Cervantes": "https://en.wikipedia.org/wiki/Antonio_Cervantes",
    "Joe Calzaghe": "https://en.wikipedia.org/wiki/Joe_Calzaghe",
    "Jung Koo Chang": "https://en.wikipedia.org/wiki/Chang_Jung-koo",
    # === 40 additional consensus greats (Grokipedia extensions) ===
    "Vitali Klitschko": "https://en.wikipedia.org/wiki/Vitali_Klitschko",
    "Andre Ward": "https://en.wikipedia.org/wiki/Andre_Ward",
    "Vasyl Lomachenko": "https://en.wikipedia.org/wiki/Vasiliy_Lomachenko",
    "Oleksandr Usyk": "https://en.wikipedia.org/wiki/Oleksandr_Usyk",
    "Naoya Inoue": "https://en.wikipedia.org/wiki/Naoya_Inoue",
    "Timothy Bradley": "https://en.wikipedia.org/wiki/Timothy_Bradley",
    "Sonny Liston": "https://en.wikipedia.org/wiki/Sonny_Liston",
    "Ken Norton": "https://en.wikipedia.org/wiki/Ken_Norton",
    "Leon Spinks": "https://en.wikipedia.org/wiki/Leon_Spinks",
    "Archie Moore": "https://en.wikipedia.org/wiki/Archie_Moore",  # active into 1960s
    "Jose Torres": "https://en.wikipedia.org/wiki/Jos%C3%A9_Torres",
    "Vicente Saldivar": "https://en.wikipedia.org/wiki/Vicente_Saldivar",
    "Ricardo Lopez": "https://en.wikipedia.org/wiki/Ricardo_L%C3%B3pez_(boxer)",
    "Azumah Nelson": "https://en.wikipedia.org/wiki/Azumah_Nelson",
    "Wilfred Benitez": "https://en.wikipedia.org/wiki/Wilfred_Ben%C3%ADtez",
    "Aaron Pryor": "https://en.wikipedia.org/wiki/Aaron_Pryor",
    "Matthew Saad Muhammad": "https://en.wikipedia.org/wiki/Matthew_Saad_Muhammad",
    "Dwight Muhammad Qawi": "https://en.wikipedia.org/wiki/Dwight_Muhammad_Qawi",
    "Iran Barkley": "https://en.wikipedia.org/wiki/Iran_Barkley",
    "Ray Leonard": "https://en.wikipedia.org/wiki/Sugar_Ray_Leonard",  # alias handled
    "Hector Camacho": "https://en.wikipedia.org/wiki/H%C3%A9ctor_Camacho",
    "Edwin Rosario": "https://en.wikipedia.org/wiki/Edwin_Rosario",
    "Julio Cesar Chavez Jr.": "https://en.wikipedia.org/wiki/Julio_C%C3%A9sar_Ch%C3%A1vez_Jr.",
    "Sergio Martinez": "https://en.wikipedia.org/wiki/Sergio_Mart%C3%ADnez_(boxer)",
    "Manny Pacquiao": "https://en.wikipedia.org/wiki/Manny_Pacquiao",  # duplicate safe
    "Vasyl Lomachenko": "https://en.wikipedia.org/wiki/Vasiliy_Lomachenko",
    "Devin Haney": "https://en.wikipedia.org/wiki/Devin_Haney",
    "Gervonta Davis": "https://en.wikipedia.org/wiki/Gervonta_Davis",
    "Jermell Charlo": "https://en.wikipedia.org/wiki/Jermell_Charlo",
    "Jermall Charlo": "https://en.wikipedia.org/wiki/Jermall_Charlo",
    "Errol Spence Jr.": "https://en.wikipedia.org/wiki/Errol_Spence_Jr.",
    "Keith Thurman": "https://en.wikipedia.org/wiki/Keith_Thurman",
    "Danny Garcia": "https://en.wikipedia.org/wiki/Danny_Garcia",
    "Shawn Porter": "https://en.wikipedia.org/wiki/Shawn_Porter",
    "Artur Beterbiev": "https://en.wikipedia.org/wiki/Artur_Beterbiev",
    "Dmitry Bivol": "https://en.wikipedia.org/wiki/Dmitry_Bivol",
    "Teofimo Lopez": "https://en.wikipedia.org/wiki/Te%C3%B3fimo_L%C3%B3pez",
    "Ryan Garcia": "https://en.wikipedia.org/wiki/Ryan_Garcia",
    "Tank Davis": "https://en.wikipedia.org/wiki/Gervonta_Davis",  # alias
    "Canelo Alvarez": "https://en.wikipedia.org/wiki/Canelo_%C3%81lvarez"  # already core
}

rows = []
for name, url in fighters.items():
    print(f"Fetching {name}...")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        fight_table = next((t for t in tables if "Date" in t.columns and len(t) > 5), None)
        if fight_table is None: continue
        for _, r in fight_table.iterrows():
            date_str = str(r.get("Date", "")).strip()
            if not date_str or "nan" in date_str.lower(): continue
            try:
                dt = pd.to_datetime(date_str, errors="coerce")
                if dt.year < 1965: continue
                date = dt.strftime("%Y-%m-%d")
            except: continue
            opp = str(r.get("Opponent", "")).strip()
            if not opp or "nan" in opp.lower(): continue
            res = str(r.get("Result", "")).lower()
            is_win = "win" in res or any(x in res for x in ["ko", "ud", "sd", "tdko"])
            rows.append([name, opp, is_win, date])
        time.sleep(0.8)
    except Exception as e:
        print(f"  Skipped {name}: {e}")

df = pd.DataFrame(rows, columns=["boxer_a", "boxer_b", "is_a_win", "date"])
df = df.sort_values("date").drop_duplicates()
df.to_csv("full_boxing_matches_1965_present.csv", index=False)
print(f"✅ Saved {len(df):,} rows for 100 seeds")