# CNN Transfer Learning for Batik Motif Classification

Project ini berisi pipeline klasifikasi citra motif batik menggunakan transfer learning berbasis CNN, kemudian model akhir dikonversi ke TensorFlow Lite untuk inferensi yang lebih ringan. Fokus utamanya adalah membangun alur end-to-end: audit dataset, pembersihan data, pemilihan kelas, training, evaluasi, konversi model, dan benchmarking artefak TFLite.

Repository ini saat ini menyimpan kode pipeline dan artefak hasil eksperimen. Folder dataset mentah `_DATASET/` sengaja tidak diunggah ke GitHub karena ukurannya besar dan hanya dipakai secara lokal.

## Ringkasan Project

- Tujuan: mengklasifikasikan motif batik Indonesia dari citra kain menggunakan transfer learning.
- Model utama: `MobileNetV2` dari `tf.keras.applications`.
- Input gambar: `224 x 224`.
- Framework: TensorFlow / Keras.
- Output akhir: model `.keras`, `SavedModel`, dan beberapa model `.tflite`.
- Varian eksekusi:
  - `batik_tflite_pipeline.py`: versi lokal dengan sumber data dari folder `_DATASET/`.
  - `batik_tflite_colab.py`: versi Google Colab dengan unduhan dataset dari Kaggle.

## Latar Belakang

Motif batik memiliki pola visual yang kaya dan sering kali mirip satu sama lain. Pendekatan klasifikasi manual sulit diskalakan, sehingga project ini memanfaatkan transfer learning agar model dapat belajar representasi visual dengan data yang relatif terbatas. Selain akurasi, project ini juga mempertimbangkan efisiensi deployment melalui TensorFlow Lite.

## Struktur Repository

```text
.
|-- README.md
|-- batik_tflite_pipeline.py
|-- batik_tflite_colab.py
|-- outputs/
|   |-- figures/
|   |-- models/
|   |-- classification_report.csv
|   |-- labels.json
|   |-- model_size_comparison.csv
|   |-- pipeline.log
|   |-- run_summary.json
|   |-- tf_vs_tflite_benchmark.csv
|   |-- train_split.csv
|   |-- val_split.csv
|   `-- test_split.csv
`-- .gitignore
```

## Dataset

Versi lokal pipeline mengasumsikan dataset berada di folder `_DATASET/`, dengan setiap subfolder mewakili satu kelas motif batik. Versi Colab menggunakan dataset Kaggle:

- `dwibudisantoso/batik-wastra-nusantara`

Pada pipeline ini, kelas dipilih menggunakan filter jumlah gambar:

- minimum gambar per kelas: `30`
- maksimum gambar per kelas: `50`

Kelas terpilih pada run yang disimpan di repository ini:

1. `motif-batik-bali`
2. `motif-batik-betawi`
3. `motif-batik-cendrawasih`
4. `motif-batik-garutan`
5. `motif-batik-gentongan`
6. `motif-batik-kawung`
7. `motif-batik-megamendung`
8. `motif-batik-priangan`
9. `motif-batik-sekar`
10. `motif-batik-semarangan`
11. `motif-batik-sogan`
12. `motif-batik-tambal`
13. `motif-batik-truntum`
14. `motif_batik_tanjungbumi`

Pipeline juga melakukan validasi file gambar. Pada run ini ditemukan 1 gambar tidak valid:

- `_DATASET\motif-batik-cendrawasih\18.jpg`

## Alur Pipeline

Secara garis besar, alur yang dijalankan oleh script adalah:

1. Membaca seluruh gambar dari dataset.
2. Menghitung jumlah data per kelas.
3. Menyaring gambar rusak atau tidak valid.
4. Memilih kelas yang jumlah citranya berada pada rentang target.
5. Membagi dataset menjadi train, validation, dan test secara stratified.
6. Membuat pipeline `tf.data` dan augmentasi data.
7. Melatih model transfer learning berbasis `MobileNetV2`.
8. Melakukan fine-tuning pada sebagian layer backbone.
9. Mengevaluasi model pada data uji.
10. Menyimpan model akhir ke format `.keras` dan `SavedModel`.
11. Mengonversi model ke beberapa varian TensorFlow Lite.
12. Membandingkan ukuran model dan latency inferensi.

## Konfigurasi Training Utama

Konfigurasi utama yang digunakan pada pipeline:

- ukuran gambar: `224 x 224`
- batch size: `16`
- seed: `42`
- training head awal: `12` epoch
- fine-tuning: `10` epoch
- jumlah layer backbone yang dibuka saat fine-tuning: `30`

## Hasil Eksperimen

Ringkasan hasil dari `outputs/run_summary.json`:

- jumlah kelas terpilih: `14`
- jumlah data train: `515`
- jumlah data validation: `64`
- jumlah data test: `65`
- test loss: `1.8749`
- test accuracy: `0.4308`
- macro F1-score: `0.4165`
- weighted F1-score: `0.4039`

Beberapa metrik per kelas dari `outputs/classification_report.csv` menunjukkan performa yang bervariasi. Contoh:

- `motif-batik-megamendung`: F1 `0.9091`
- `motif-batik-kawung`: F1 `0.8333`
- `motif-batik-semarangan`: F1 `0.7500`
- beberapa kelas masih lemah, misalnya `motif-batik-bali` dan `motif-batik-tambal`

Ini menunjukkan bahwa dataset masih menantang, baik karena jumlah sampel per kelas terbatas maupun kemiripan visual antar motif.

## Artefak Model

Folder `outputs/models/` berisi model hasil training:

- `best_head.keras`
- `final_model.keras`
- `saved_model/`
- `model_dynamic.tflite`
- `model_float16.tflite`
- `model_int8.tflite`

Perbandingan ukuran model dari `outputs/model_size_comparison.csv`:

- `SavedModel`: `19.04 MB`
- `TFLite Dynamic`: `2.41 MB`
- `TFLite Float16`: `4.29 MB`
- `TFLite Int8`: `2.60 MB`

## Benchmark TensorFlow vs TFLite

Berdasarkan `outputs/tf_vs_tflite_benchmark.csv`, hasil benchmarking pada run ini adalah:

- TensorFlow: akurasi `0.4615`, latency `205.52 ms`
- TFLite Dynamic: akurasi `0.4462`, latency `60.42 ms`
- TFLite Float16: akurasi `0.4615`, latency `28.60 ms`
- TFLite Int8: akurasi `0.4154`, latency `14.87 ms`

Interpretasi singkat:

- model `float16` memberi trade-off terbaik pada run ini karena akurasi setara TensorFlow dengan latency jauh lebih rendah
- model `int8` paling cepat, tetapi mengalami penurunan akurasi
- model `dynamic` tetap jauh lebih ringan dan cukup dekat performanya

## Visualisasi yang Tersedia

Folder `outputs/figures/` berisi visualisasi yang mendukung analisis pipeline, antara lain:

- distribusi kelas sebelum dan sesudah seleksi
- contoh gambar per kelas
- demo preprocessing
- distribusi split train/val/test
- demo augmentasi
- status backbone pada fase head dan fine-tuning
- kurva training
- confusion matrix
- heatmap classification report
- contoh prediksi yang salah
- perbandingan ukuran model
- benchmark TensorFlow vs TFLite

Artefak ini berguna untuk laporan penelitian, dokumentasi eksperimen, dan evaluasi model.

## Cara Menjalankan

### 1. Menjalankan versi lokal

Pastikan dataset tersedia di folder `_DATASET/`, lalu install dependensi Python yang relevan, misalnya:

```bash
pip install tensorflow pandas numpy matplotlib seaborn pillow scikit-learn
```

Kemudian jalankan:

```bash
python batik_tflite_pipeline.py
```

### 2. Menjalankan versi Google Colab

Gunakan file:

```bash
batik_tflite_colab.py
```

Script ini sudah menyiapkan instalasi package tambahan yang dibutuhkan di Colab dan akan mencoba mengunduh dataset dari Kaggle.

## Kegunaan Project

Project ini cocok dijadikan dasar untuk:

- penelitian klasifikasi citra batik
- eksperimen transfer learning pada dataset budaya lokal
- pembuatan model ringan untuk deployment mobile atau edge device
- perbandingan format model TensorFlow dan TensorFlow Lite
- dokumentasi workflow ML dari data audit sampai benchmarking

## Keterbatasan Saat Ini

- akurasi keseluruhan masih moderat, belum cukup untuk deployment produksi tanpa iterasi lanjutan
- beberapa kelas masih sulit dibedakan
- jumlah sampel per kelas relatif kecil
- dataset mentah tidak ikut repository, sehingga reproduksi lokal memerlukan data source terpisah

## Pengembangan Lanjutan

Beberapa arah peningkatan yang masuk akal:

- menambah jumlah data per kelas
- melakukan balancing atau augmentasi yang lebih agresif
- mencoba backbone lain seperti EfficientNet
- melakukan hyperparameter tuning
- mengevaluasi confusion pairs antar motif yang paling mirip
- menyiapkan inference script terpisah untuk deployment

## Status Repository

Repository GitHub ini menyimpan:

- source code pipeline
- output eksperimen
- model hasil training
- dokumentasi project

Folder `_DATASET/` tidak diunggah agar repository tetap lebih ringan dan fokus pada artefak yang relevan untuk studi, evaluasi, dan reproduksi pipeline.
