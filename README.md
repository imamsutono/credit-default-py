# Customer loan product

Meningkatkan kinerja customer loan product dengan melakukan:<br>
4a. [Descriptive analysis](https://github.com/imamsutono/credit-default-py/blob/master/descriptive_analysis.py)<br>
4b. [Statistical significant test dengan confidence 95% untuk mencari atribut yang mempengaruhi default](https://github.com/imamsutono/credit-default-py/blob/master/default_determinant.py)<br>
5. [Classification model](https://github.com/imamsutono/credit-default-py/blob/master/classification_model.py)<br>

### Langkah-langkah hyperparameter tuning:
1. **Pemilihan Hyperparameter yang Akan Di-Tune**
Menentukan hyperparameter yang ingin di-tune. Dalam Regresi Logistik, beberapa hyperparameter yang dapat diatur adalah `C` (inverse of regularization strength), `penalty` (jenis regularisasi), dan lain-lain.
2. **Pembuatan Grid Search atau Random Search**<br>
Menggunakan Grid Search untuk mencari kombinasi nilai hyperparameter yang optimal dengan mencoba semua kombinasi dari nilai yang didefinisikan.
3. **Definisikan Kombinasi Hyperparameter**<br>
Menentukan kombinasi nilai hyperparameter yang akan diuji.
4. **Pelatihan dan Evaluasi**<br>
Latih model pada data pelatihan dan evaluasi performanya pada data pengujian menggunakan metrik yang sesuai.
5. **Pilih Hyperparameter Terbaik**<br>
Menentukan kombinasi hyperparameter yang memberikan performa terbaik pada data pengujian.