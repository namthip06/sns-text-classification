# Text Classification 18 หมวด — สรุปบทสัมภาษณ์

> วันที่: 2026-08-27 · บทสัมภาษณ์ (interview-me) ก่อนเริ่ม implement

## โจทย์

- ต้องการ **local model** จำแนกข้อความ Twitter ไทย เข้า **18 หมวด** (หมวดเนื้อหา)
- แต่ละโพสต์ได้ **หมวดเดียว** (single-label)
- ข้อมูล 1 แสน post **ไม่มี label**
- ข้อความมี **hashtag** ในเนื้อหา แต่ **ไว้ใจไม่ได้** (ติดมั่ว, บางโพสต์ไม่มี tag)
- ภาษาไทยแบบเรียลๆ: **เสียดสี / สะกดผิดตั้งใจเลี่ยงคำ / พูดเป็นนัยๆ** → keyword match ใช้ไม่ได้
- ปลายทาง: predict หมวด + **ค่า confidence** ไปแสดงผลต่อ
- มี **GPU** · มีงบจ่าย API (Claude/GPT) ช่วยสร้าง label แต่ต้องควบคุมปริมาณเพราะ **class imbalance**

## ข้อสรุปหลัก

1. **ใช้ unlabeled 100k กับ label (จาก LLM) ต่างบทบาทกัน** — ไม่ใช่เลือกอย่างใดอย่างหนึ่ง
2. **continue MLM pretraining มีค่า แต่ไม่ใช่ตัวชี้ขาด** — เพราะ WangchanBERTa/PhayaThaiBERT เห็นข้อมูลโซเชียลไทยมาแล้ว (wisesight-large ~51GB)
3. **self-training เป็นแค่ตัวเสริมท้ายๆ** — teacher (LLM) เก่งกว่า student อยู่แล้ว เสี่ยง error propagation

## แนวทางที่แนะนำ

### Flow รวม (กรณี 2 → กรณี 1)

สองกรณีไม่ใช่ทางเลือกคู่ขนาน — มันคือ **สองช่วงของ flow เดียว**: กรณี unlabeled คือ "สร้าง label ก่อน" แล้วไหลเข้าสู่กรณี labeled

```
                       ┌───────────── กรณีที่ 2 : unlabeled ─────────────┐      ┌───────────── กรณีที่ 1 : labeled ─────────────┐
                       ▼                                                 │      ▼                                               │
[100k unlabeled] → Preprocess → Weak label (tag rule) → LLM label → Merge ~10-20k → Split → Fine-tune → Confidence → Eval → Deploy
                 (เก็บ #)  ↑                              ↑               │                        │
                           │                              └─ sample เน้นคลาสหายาก / โพสต์ยาก      │
                           └─ cover คลาสเยอะ ฟรี            (ไม่มี tag / เสียดสี / เลี่ยงคำ)        │
                                                                                                  ▼
                                                                              test ใช้ชุดมนุษย์ label จริง ~1-2k
```

ย่อเป็นภาพเดียว:

```
[กรณี 2] สร้าง label (weak rule + LLM) ──► [กรณี 1] fine-tune + eval + deploy
```

### Flow แบบย่อ (แบบเส้นเดียว)

```
[100k unlabeled] → Preprocess → Weak label (tag rule) → LLM label → Merge ~10-20k → Fine-tune → Eval → Deploy
```

### ขั้นสร้าง label (กรณี unlabeled)

| ขั้น | วิธี | หมายเหตุ |
|---|---|---|
| Weak label | rule จาก hashtag ที่ชี้ชัด → label ฟรี | จุดประสงค์หลักคือ **จัดกลุ่มคร่าวๆ เพื่อประหยัดงบ LLM** |
| LLM label | Claude/GPT + คำอธิบาย 18 หมวด + few-shot, ตอบเป็น JSON, temperature ~0.2 | เน้นคลาสหายาก + โพสต์ไม่มี tag + โพสต์ยาก (เสียดสี/เลี่ยงคำ) |
| ควบคุม cost | สุ่ม label ~2-3k ก่อนวัดการกระจายจริง → จัดสรรงบตามคลาส (หายากเจาะด้วย tag hint) | แก้ปัญหาคลาสหายาก 1-2% ต้อง scan เยอะถ้าไม่มี hint |

### ขั้นเทรน (กรณี labeled — ใช้กับทุกกรณี)

| ขั้น | รายละเอียด |
|---|---|
| Preprocess | ตัด URL/@, ตัวซ้ำ → `[CREP]`, `555` → `[LAUGH]`, ตัด `#` เก็บคำ |
| Split | stratified ตาม label · **test ต้องเป็นชุดที่มนุษย์ label จริง ~1-2k** |
| Fine-tune | PhayaThaiBERT, CE + class weight (inverse frequency), lr ~1e-5–3e-5 |
| Confidence | softmax prob → temperature scaling → threshold ต่ำกว่านั้นขึ้น "low confidence" |
| Eval | **macro-F1** + per-class recall + confusion matrix |
| Deploy | ONNX/TorchScript + quantization → inference local |

## ตัวเลือกโมเดล pretrained

| โมเดล | จุดเด่น | ข้อควรรู้ |
|---|---|---|
| **PhayaThaiBERT** (แนะนำ) | เก่งไทย, vocab ใหญ่ขึ้นจาก XLM-R, ชนะหลายงานโซเชียลไทย | เทรนต่อจาก WangchanBERTa ข้อมูลใหญ่กว่า |
| WangchanBERTa | เทรนด้วยโซเชียลไทย 51GB (wisesight-large) อยู่แล้ว | เป็นฐานของ PhayaThaiBERT |
| TwHIN-BERT | Twitter เอง เทรนด้วยทวีตจริง รวมไทย | BERT หลายภาษาเก่า, แลกความเชี่ยวชาญไทย |

## การใช้ unlabeled 100k (จัดลำดับค่า)

1. **Continue MLM pretraining** — ปรับภาษาให้เข้ากับทวิตช่วงนี้ (ถูกสุด/คุ้มสุด) แต่ gain ลดลงเพราะโมเดลเห็นโซเชียลมาแล้ว
2. **Self-training / pseudo-label** — เอาเฉพาะ confidence สูง, เป็นตัวเสริม ไม่ใช่ตัวหลัก

## ยังค้างอยู่ / ขั้นต่อไป

- [ ] ออกแบบ prompt สำหรับ LLM labeling ให้ละเอียด (few-shot ต่อหมวด, format JSON)
- [ ] ตัดสินใจขนาด label set สุดท้าย (~10-20k?) + กลยุทธ์สุ่มเน้นคลาสหายาก
- [ ] จัดทำชุดมนุษย์ label สำหรับ eval (~1-2k)
- [ ] ตั้ง baseline: PhayaThaiBERT แบบ supervised ล้วน ก่อนค่อยเพิ่ม optional continue MLM
