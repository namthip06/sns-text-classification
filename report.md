# ควรเพิ่ม `no_match` เป็นคลาสที่ 19 ใน train data หรือไม่?

*สืบค้นเมื่อ 2026-09-22 · บริบท: classifier ทวิตไทย 18 หมวด ปัจจุบัน no_match เกิดจาก G4 (conf < 0.6 หลัง temperature scaling) เท่านั้น — โมเดลไม่เคยเห็น no_match ตอน train*

---

## สรุปผู้บริหาร

**ควรทำ ถ้าหาข้อมูล train ของ no_match ได้เพียงพอและ代表ตัวจริง (representative) — งานวิจัยสนับสนุนชัดเจนว่า explicit OOS class ชนะ threshold-only** แต่ถ้าทำ no_match data ไม่ได้ ควรคง threshold-based (G4) ไว้ เพราะเป็น baseline มาตรฐานที่ไม่ต้องใช้ข้อมูล OOS เลย

---

## หลักฐานจากงานวิจัย

### สนับสนุน "เพิ่มคลาส no_match"

- **[Larson et al., EMNLP 2019](https://aclanthology.org/D19-1131/)** — กระดาษกระดาษตั้งต้นของสายนี้ (Banking-OOS / CLINC150, 800+ citations) เทียบ 3 แบบ:
  1. `oos-threshold` — ตัดด้วย max softmax prob (แบบ G4 ปัจจุบัน)
  2. `oos-binary` — binary classifier แยก in/out-of-scope
  3. `oos-train` — เพิ่ม OOS เป็นคลาสใน train (**แบบคลาสที่ 19**)

  พบว่า **แบบเทรน OOS เป็นคลาสตรง + threshold ทำงานได้ดีกว่า threshold เดี่ยว** — และ threshold-only พลาด "hard OOS" ที่ใกล้เคียงคลาสจริง (โดนให้ confidence สูงผิด ๆ)

- **[Zawbaa et al., 2024](https://aclanthology.org/)** — ยืนยันว่า "intent classifier ที่ดีต้องทำ 2 อย่าง: จำแนก intent + จับ OOS" และยังใช้แนว Larson เป็น baseline

- **มาตรฐานอุตสาหกรรม**: production chatbot (เช่น Rasa fallback intent) ใช้ explicit OOS class เป็น default

### ข้อควรระวังของ "เพิ่มคลาส no_match"

- **[Multi-cluster Boundary Learning (arXiv)](https://arxiv.org)** — accuracy ของแบบ trained-rejection-class **ตกเมื่อจำนวนคลาสโต** — 18 คลาส + no_match อยู่ในช่วงเริ่มเสี่ยง
- **[SCOOS, EMNLP Findings 2024](https://aclanthology.org/2024.findings-emnlp.531.pdf)** — แสดงว่ายังมีที่ว่างให้ปรับปรุงจาก threshold-based อีกมาก (บน Banking: +34.66% จาก MSP)
- **[Class-imbalanced + noisy label review, 2025](https://www.sciencedirect.com)** — **catch-all class มักกลืนขยะ heterogeneous + label เสียงรบกวน** — ทำให้ no_match เป็น "ถังขยะ" ที่ทำให้โมเดลสับสนได้ ถ้า weak rule ของ no_match หยาบ
- **[Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)** / **[imbalanced-learn pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html)** — ปัญหา imbalance มาตรฐาน: ถ้า no_match มีข้อมูลน้อยมาก (เช่นเดียวกับ `religion`/`child_sexual_content` ที่ weight = 0 อยู่แล้ว) จะไม่ช่วยอะไร

### สนับสนุน "ใช้ threshold ต่อไป"

- Threshold-based (MSP) ยังเป็น baseline ที่แข็งแรงเมื่อเทียบกับความยุ่งยากของการเก็บ OOS data
- **แนวปฏิบัติสมัยใหม่: ทำทั้งคู่** — เทรน explicit OOS class **แล้ว** คง calibrated threshold ครอบอีกชั้น (secondary safeguard)

---

## เทียบกับสถานการณ์โปรเจกต์นี้

| ประเด็น | สถานการณ์ปัจจุบัน | ผลกระทบถ้าเพิ่มคลาสที่ 19 |
|---|---|---|
| ที่มาข้อมูล | weak label (tag rule) 16 → 18 หมวด | ต้องมี weak rule สำหรับ no_match **ที่ยังไม่มีอยู่** — ต้องออกแบบ rule ใหม่ (เช่น rule ที่จับ "คุยเฉย ๆ / ไม่เกี่ยวข้อง") |
| Ground truth | ไม่มีมนุษย์ label (decision 2026-08-27) | no_match จาก weak rule อาจเป็น "ถังขยะ" — เสี่ยง label noise สูงกว่าคลาสอื่น |
| G3 ปัจจุบัน | no_match F1 = 0 เสมอ (true ไม่มี no_match) | เมตริกจะวัด recall ของ no_match ได้จริง — วัดสิ่งที่วัดไม่ได้ตอนนี้ |
| G4 threshold | ยังจำเป็นอยู่ดี | **ไม่ควรถอด** — งานวิจัยแนะให้ทำคู่กัน (OOS class + calibrated threshold) |
| คลาสที่ไม่มีข้อมูล | `religion`, `child_sexual_content` weight = 0 | กลายเป็นคลาสที่ 3 ที่ต้องระวังเรื่อง imbalance |
| ผู้ใช้ปลายทาง | predict → `low_confidence` column | no_match จะเป็นผลลัพธ์ที่ "โมเดลเชื่อจริง" ไม่ใช่แค่ "ไม่มั่นใจ" — ตีความต่างกัน |

---

## ข้อเสนอแนะ

1. **เงื่อนไขที่ควรทำ**: มี weak rule ของ no_match ที่จับ "ข้อความนอก 18 หมวด" ได้จริง + ปริมาณเพียงพอ (อย่างน้อยใกล้เคียงคลาสที่เล็กที่สุดใน 18) + สุ่มตรวจตัวอย่างด้วยตาว่า rule ไม่กลืนขยะ
2. **เงื่อนไขที่ไม่ควรทำ**: ถ้าต้องสร้าง no_match จาก "แถวที่ rule อื่นไม่จับ" ล้วน ๆ — นั่นคือวงจรเลี้ยวตาม (circular) และเสี่ยงเป็นถังขยะตามที่งาน review เตือน
3. **แนวทางระหว่างทาง (แนะนำ)**: ใช้ `data/raw_posts.csv` (258k แถว ยังไม่ใช้) สุ่มตัวอย่างที่ไม่โดน weak rule ใด → ตรวจด้วยตา/LLM ว่าเป็น no_match จริงกี่ % — ถ้าสูงพอ นี่คือ weak source ของ no_match โดยไม่ต้องเขียน rule ใหม่ *หมายเหตุ: ต้องถามอนุมัติก่อนตาม CLAUDE.md (เปิด preprocess/LLM จาก raw)*
4. **ไม่ว่าทางไหน คง G4 threshold ไว้** — งานวิจัยแนะทำสองชั้นคู่กัน ไม่ใช่เลือกอย่างเดียว

---

## แหล่งอ้างอิง

- [Larson et al. (2019) — An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://aclanthology.org/D19-1131/)
- [Uncertainty Estimation for Open-Set Text Classification (arXiv)](https://arxiv.org/html/2604.08560v1)
- [SCOOS: Class Name Guided Out-of-Scope Intent Classification (EMNLP Findings 2024)](https://aclanthology.org/2024.findings-emnlp.531.pdf)
- [Improved Out-of-Scope Intent Classification with Dual... (Zawbaa et al., 2024)](https://aclanthology.org)
- [Multi-cluster Boundary Learning for OOS Intent Detection (arXiv)](https://arxiv.org)
- [Open-world Machine Learning (ACM Computing Surveys 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3561381)
- [Text Classification Under Class Distribution Shift: A Survey (arXiv)](https://arxiv.org/html/2502.12965v3)
- [Imbalanced classification with label noise: A systematic review (2025)](https://www.sciencedirect.com)
- [Google ML Crash Course — Imbalanced Datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)
- [imbalanced-learn — Common Pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [CLINC150 OOS dataset (Kaggle)](https://www.kaggle.com)

*หมายเหตุ: ลิงก์ ACL Anthology / arXiv บางรายการมาจากผลค้นหาแบบสรุป แนะให้คลิกตรวจสอบก่อนอ้างในเอกสารทางการ*
