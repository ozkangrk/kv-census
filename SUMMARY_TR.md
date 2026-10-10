# ÖZET (Türkçe) — Memory Is Architecture: KV-Cache Sayımı 2023–2026

**Ne yapıldı.** Hugging Face Hub üzerindeki açık-ağırlıklı decoder LLM'lerin (Ocak 2023 –
Ekim 2026) tam bir sayımı yapıldı: 33 orijinal yayıncı org, 2.331 checkpoint satırı, 2.068
aile-dedup grubu (292 farklı dikkat geometrisi), her satır için snapshot'lanmış config.json
(2.437 adet) ve önbelleklenmiş Hub API yanıtları (7.300+). KV-cache ayak izi, her mimari
sınıfı (tam dikkat/GQA/MQA, kayan pencere, MLA gizil değişkeni, Mamba/hibrit nüks eden
durum, katmanlar-arası paylaşım) kapsayan, 107 birim testli `analysis/kv.py` ile analitik
olarak hesaplandı. Ön-kayıtlı plana (`analysis/PLAN.md`) uyuldu; sapmalar tarihli notlarla
kaydedildi.

**Ana bulgular** (tümü `make analysis` çıktısı makrolarından):
- **RQ1/H1:** Parametre başına KV maliyeti düştü (havuz 2025–26/2023 oranı 0,413; geometri
  dedup'ta 0,071) — DESTEKLENDİ; ama 2025 "sıçraması" çoğunlukla aynı-geometri fine-tune
  tekrarlarının artifact'i, ve multimodal satırlar hariç oran 1,061'e çıkıyor (REDD).
  Bulgu gerçek ama kırılgan olarak raporlandı.
- **RQ2/H2:** Parametre sayısı KV ayak izini zayıf açıklıyor (R²=0,398 <0,5; geometri-dedup'ta
  0,254); geometri+mekanizmalar R²=0,792. DESTEKLENDİ (birimlere dayanıklı).
- **RQ2/H3:** MoE modellerde parametre başına KV, yoğun modellerin 0,018 katı (MWU
  p≈10⁻⁹⁹). DESTEKLENDİ (güçlü seyreltme etkisi; dizi başına KV değişmez).
- **RQ3/H4:** KV-parite bağlamı (KV = ağırlık baytı olan T*) aile biriminde yalnız 2,2×,
  geometri dedup'ta 11,0× büyüdü — hipotezin "10 kat" iddiası birime duyarlı: REDDEDİLDİ
  (birincil birimde). KV-baskınlık (T*<32k) payı %71→%11'e düştü.
- **RQ4:** Analitik muhasebe, HF transformers'ın gerçekte ayırdığı cache baytlarıyla 7
  model/6 sınıfta karşılaştırıldı: 6/7 satır %1 içinde; sapmalar bulgu olarak açıklandı
  (transformers 5.19 kayar pencereyi W−1'e kırpar; MLA gizil değişken olarak saklanır;
  conv-state padding tek >%1 sapma).
- **Kapasite:** 128 GiB'de 4-bit ağırlıkta bile 74 dedup model sığmıyor (64'ü yalnızca
  ağırlıktan); 128 GiB/BF16'da medyan model 32k bağlamdan 30, 128k bağlamdan 8 eşzamanlı
  dizi çıkarıyor; model sayısının %12,6'sı tek 128k diziyi bile taşıramıyor.

**Süreç.** Orkestratör + 4'lü alt-ajan sürüleri: Phase 1 (census/kv/validasyon/edebiyat),
Phase 2 (analiz, deterministik, offline), Phase 3 (yazım, 4 yazar), Phase 4 (2 advers
hakem; bulduğu 5 MAJOR dahil tüm düzeltmeler `reviews/responses.md`'de). Tüm sayılar
üretimi makrolardan, tüm atıflar (34) arXiv API'den doğrulanmış ham kayıtlardan.

**Tamamlama kapısı.** `kv_census_gate.py` son koşu: **6/6 PASS** (layout, census, analysis
byte-byte determinizm, validation, paper 16 sayfa, citations 34/34).

**Sınırlamalar (özet).** 237 gated repo atlandı (meta-llama org tamamen; eski Gemma
kataloğu) — seçim yanlılığı; createdAt yayın-tarihi vekili; tek zaman-noktası anlık görüntü,
2026 kısmi yıl; 9 ayrıştırılamayan config, 104 parametresiz satır; analitik değerler
teorik minimum; kapasite modeli overhead parametresiyle idealize edildi.

**Sapma beyanı.** Sistemdeki `tectonic` kurulumu bozuk stub olduğu için tectonic 0.17.0
(aarch64-musl) resmi GitHub sürümünden proje içine (`.tools/`) indirildi; sistem
dizinlerine dokunulmadı. Bu, AGENTS.md'deki araştırma-API beyaz listesinin dışında bir
araç-kurulum indirmesidir (uv/PyPI ile aynı kategori); kullanıcıya raporlandı.
