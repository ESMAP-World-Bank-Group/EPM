

# Electricity Planning Models Comparison

| Criteria                        | **EPM**                        | **PyPSA**                    | **PLEXOS**                    | **PSR (OptGen)**             | **ANTARES**                   |
|--------------------------------|--------------------------------|------------------------------|-------------------------------|------------------------------|-------------------------------|
| **Dispatch & Capacity Expansion** | ✔ Both                        | ✔ Both                        | ✔ Both                         | ✔ Both                        | ✔ Dispatch only               |
| **Open-Source**                | ✔ Model only                  | ✔ Fully                      | ✘                             | ✘                            | ✔                             |
| **Cost**                       | $5k–$10k (GAMS license)        | $0–$20k (solver only)        | ~$75k/user/year               | $30k–$100k+                  | Free (SaaS operational)       |
| **Granularity**                | Hourly, National/Regional      | Hourly, National/Regional    | Sub-hourly, Regional/Global   | Hourly, National             | Hourly                        |
| **Language**                   | GAMS                           | Python                       | C++, Java, Python             | GAMS                         | —                             |
| **Detail**                     | Custom constraints/params; Excel, CSV, Python compatible | Modular Python-based; high user control | Rich object library; multi-energy support | Standardized structure; moderate adaptability | Flow stress simulation; large input capacity |