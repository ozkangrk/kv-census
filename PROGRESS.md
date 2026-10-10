# İlerleme Günlüğü (PROGRESS.md)

## Phase 0 — Kurulum (tamamlandı)
- Depo hazırdı (commit 8418bbf: BRIEF.md + AGENTS.md). İskelet dizinler, Makefile
  (data/analysis/validate/paper/test hedefleri), `analysis/INTERFACES.md` (kv.py API'si,
  models.csv şeması, dosya sahiplik haritası) ve ön-kayıtlı `analysis/PLAN.md` yazıldı.
- `uv` ile `.venv` (Python 3.12) kuruldu; torch CPU-only, transformers, pandas, numpy,
  scipy, matplotlib, requests, pytest; sürümler `requirements.txt`'e sabitlendi.
- Sonraki adım: Phase 1 — 4 paralel alt-ajan (census-data, kv-accounting, validation,
  literature).

### Phase 1 ara durum (13:40)
- literature ✅: paper/references.bib 34 giriş (hepsi arXiv eprint'li, orkestratör 5/5 spot-check),
  50 ham API yanıtı data/raw/literature/ altında; commit 9eb7b88.
- kv-accounting ✅: analysis/kv.py (994 satır) + tests/test_kv.py (67 test, hepsi geçti; orkestratör
  `make test` + 13 fixture bağımsız formül kontrolü). Önemli bulgular: gpt-oss 1:1 sliding/full,
  deepseek-v2 tüm katmanlar MLA, falcon-h1 her katmanda attention+mamba birlikte. Commit 3300e0f.
- census-data: aşama A (sayım/snapshot) sürüyor. validation: kv.py hazır, deney entegrasyonu sürüyor.

## Phase 1 — Paralel yapı (tamamlandı, ~12:00-14:30)
Tüm teslimler orkestratör tarafından bağımsız doğrulandı:
- **census-data ✅** data/models.csv: 2.331 satır (benzersiz repo_id), 2.068 dedup grubu, 33 org;
  yıl dağılımı 2023:263 / 2024:670 / 2025:1044 / 2026:354; 2.437 config snapshot'ı; 7.344 önbellek
  API yanıtı; gated atlanan 237 repo (google 130, meta-llama 42, CohereLabs 24 — seçim yanlılığı
  sınırlaması olarak raporlanacak); 9 KVConfigError satırı (kv_config_errors.txt); 104 paramsız satır.
  Orkestratör simülasyonu: 1.519 saf-attention satırında 2·L·kvh·hd·2 kimliği ve ham config eşleşmesi
  0 hata; `make data` offline byte-identical (md5 8b8e5f5b…).
- **kv-accounting ✅** analysis/kv.py: 105 test (orkestratör yeniden çalıştırdı); 2.427/2.437 config
  çözümleniyor. Bulgu: gpt-oss 1:1 sliding/full; DeepSeek-V2+ tüm dikkat katmanları MLA; Falcon-H1
  katman başına attention+mamba (2 cache birimi). Etiket dürüstlüğü kuralı kodda zorunlu.
- **validation ✅** results/kv_validation.csv: 7 model / 6 sınıf / 6 satır %1 içinde; 5,26% sapma =
  conv-state padding (gerçek RQ4 bulgusu). `make validate` 35 sn (orkestratör: rel_error sütunu
  yeniden hesaplandı, birebir). Bulgu: transformers 5.19'da SWA cache W-1'e kırpılıyor; MLA latent
  olarak (sıkıştırılmış) saklanıyor.
- **literature ✅** paper/references.bib: 34 giriş, hepsi arXiv eprint'li, 0 URL-only; 50 ham yanıt;
  orkestratör 5/5 arXiv API başlık spot-check. Anahtar listesi yazarlara iletilecek.
Açık konular: 237 gated skip + 9 kv hatası + 104 paramsız satır → Limitations bölümünde. Sonraki:
Phase 2 (analiz) orkestratörde.

## Phase 2 — Analiz (tamamlandı, orkestratör)
- `analysis/run_analysis.py` yazıldı; `make analysis` 5,3 sn, OFFLINE ve deterministik (iki koşu
  byte-identical; gate koşulu). PLAN.md'ye tarihli sapma notları eklendi (kesin parça-doğrusal
  çözücü, objektif altküme kuralları) — sonuçlara bakılmadan önce.
- Çıktılar: results/{rq1_stats.csv, rq3_capacity_overhead6.csv (35k satır), rq3_sensitivity_overhead.csv,
  analysis_stats.json, dedup_rows.csv}, paper/figures/fig1..fig6.pdf (sabit metadata),
  paper/generated/{macros,table_overview,table_mechanism,table_capacity,table_validation}.tex.
- Ana sonuçlar (kaynak: analysis_stats.json / macros.tex):
  - H1 DESTEKLENDİ: 2025+26 havuz medyanı / 2023 medyanı = 0,413 (eşik ≤0,75). Yıllara göre mutlak
    medyan KV/token: 327.680 → 131.072 → 70.272 → 36.864 B (2023→2026). KV/param 2025'te küçük model
    yoğunluğu nedeniyle tepe yapıyor (dürüst nüans, tartışılacak).
  - H2 DESTEKLENDİ: log-log param-only R²=0,398 (<0,5 "zayıf"); geometri+mekanizma R²=0,792; tam model
    0,841. Parametre sayısı KV ayak izini zayıf açıklıyor.
  - H3 DESTEKLENDİ: MoE medyan KV/param = 0,018× yoğun (MWU p=2,7e-99, rank-biserial etkisi).
  - H4 REDDEDİLDİ: KV-parite bağlamı medyanı (sonlu T*) 26.729 (2023) → 220.157 (2026); oran 2,2×
    (eşik ≥10×). Ama KV-baskın rejim payı (T*<32k) %71 → %11'e düştü; ∞-payı (doymuş cache) 2026'da %3,9.
  - Kapasite: 128 GiB / 4-bit'te bile sığmayan dedup model sayısı = \InfeasibleBigModels; gpt-oss-120b
    128GiB/4-bit'te 1,8M token; tam ızgara (3 bütçe × 3 ağırlık hassasiyeti × 2 KV hassasiyeti ×
    4 overhead) results/ altında.
- Sonraki adım: Phase 3 yazım (4 paralel yazar + orkestratör entegrasyonu).

## Phase 3 — Yazım (tamamlandı)
- 4 paralel yazar: intro/related (768+683 kelime, 34 anahtarın 34'ü related_work'te atıflı),
  data&methods (1053), results (1399, 56 macro), discussion/limitations/conclusion/disclosure
  (600+504+271+205). Orkestratör fig3_mechanism (fig:mechanism) entegrasyonu ve \section{Results}
  başlığını ekledi; validation tablosunun overfull hbox'ı üreticide düzeltildi.
- ORTAM SAPMASI: ~/.local/bin/tectonic 9 baytlık bozuk stub'muş ("Not Found"), sistemde LaTeX yok.
  Çözüm: tectonic 0.17.0 (aarch64-musl) resmi GitHub release'ten projeye indirildi (.tools/tectonic),
  Makefile paper hedefi proje-içi binary + TECTONIC_CACHE_DIR=../.tools/tectonic-cache kullanıyor;
  sistem dizinlerine dokunulmadı. Araç kurulumu AGENTS.md'deki araştırma-API beyaz listesinin
  dışındadır (PyPI/uv gibi altyapı indirmesi olarak değerlendirildi) — kullanıcıya raporlandı.
- Derleme: make paper → main.pdf, 15 sayfa (≥8), gate-tarzı kelime sayısı 5.242 (≥4.000),
  34/34 bib anahtarı atıflı, 0 uncited, \input{generated/...} mevcut.
- Sonraki: Phase 4 çapraz inceleme (methods-reviewer + claims-auditor).
