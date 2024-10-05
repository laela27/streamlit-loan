import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV


# Load dataset
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9i0KlgP1WY52MFh55WxF1bAeoXWUu4a5z3OV-DplxlRRyIEvZxLa58UbUox1W6nlzYEYZw9AF94z-/pub?output=csv')

# Print Judul
st.title("Loan Approval Prediction")

# Project Overview
st.subheader("A. Project Overview")
st.write("**a. Background**: Industri keuangan menghadapi tantangan dalam menilai kelayakan peminjam di tengah meningkatnya aplikasi pinjaman. Diperlukan sistem yang memberikan penilaian cepat dan tepat untuk meminimalisir risiko kredit dan memaksimalkan profitabilitas.")
st.write("**b. Problem**: Penentuan status persetujuan pinjaman masih sering dilakukan secara manual, yang dapat menyebabkan proses yang lambat dan keputusan yang kurang akurat.")
st.write("**c. Goal**: Membangun model prediksi persetujuan pinjaman yang dapat memberikan hasil secara otomatis dan akurat, sehingga membantu lembaga keuangan dalam pengambilan keputusan.")
st.write("**d. Objective**: Menganalisis faktor-faktor yang mempengaruhi status pinjaman, membangun model prediksi akurat, dan mengevaluasi kinerjanya untuk perbaikan proses persetujuan pinjaman.")

# Data Understanding
st.subheader("B. Data Understanding")
st.write("**a.** **Terdapat 4269 baris dan 13 kolom.**")
st.write("**b.** **Terdapat 12 features**")
st.write("**c.** **Terdapat 1 Target (loan_status).**")

# Deskripsi Kolom Data Set
st.write("**d. Dekripsi kolom data set.**")

# Membuat DataFrame untuk tabel deskripsi kolom
data_columns = {
    "No": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "Nama Kolom": [
        "loan_id", "no_of_dependents", "education", "self_employed", 
        "income_annum", "loan_amount", "loan_term", "cibil_score", 
        "residential_assets_value", "commercial_assets_value", 
        "luxury_assets_value", "bank_asset_value", "loan_status"
    ],
    "Deskripsi": [
        "Identifikasi unik untuk setiap pinjaman.",
        "Jumlah tanggungan.",
        "Status pendidikan applicant.",
        "Status pekerjaan applicant, apakah bekerja sendiri atau tidak.",
        "Pendapatan tahunan applicant.",
        "Jumlah pinjaman yang diminta.",
        "Jangka waktu pinjaman.",
        "Skor CIBIL applicant.",
        "Nilai aset residensial.",
        "Nilai aset komersial.",
        "Nilai aset mewah.",
        "Nilai aset.",
        "Status pinjaman (Approved atau Rejected)."
    ]
}

# Mengubah dictionary menjadi DataFrame
df_columns = pd.DataFrame(data_columns)

# Menampilkan tabel deskripsi kolom di Streamlit
st.table(df_columns)


# Exploratory Data Analysis (EDA)
st.subheader("C. Exploratory Data Analysis (EDA)")

# Menampilkan 5 baris pertama dari DataFrame
st.write("**a. DataFrame Preview:**")
st.dataframe(df.head())

# Menampilkan 10 nilai unik pertama untuk setiap kolom
st.write("**b. Nilai Unik dari Setiap Kolom:**")
for column in df.columns:
    st.write(f"{column}: {df[column].unique()[:10]}")  # Menampilkan hingga 10 nilai unik pertama dalam setiap kolom

# Membersihkan nama kolom jika ada spasi ekstra
df.columns = df.columns.str.strip()

# Memastikan tidak ada spasi ekstra di nilai loan_status
df['loan_status'] = df['loan_status'].str.strip()

# Memastikan tidak ada spasi ekstra di nilai education dan self_employed
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()


# Menampilkan Distribusi Data
st.write("**c. Distribusi Data:**")

# Group column names based on type
categoricals = ['education', 'self_employed', 'loan_status']
numericals = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
              'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
              'luxury_assets_value', 'bank_asset_value']

# Pilih jenis visualisasi
visualization_type = st.selectbox("Pilih jenis visualisasi:", ("Distribusi Kategorikal", "Distribusi Numerik"))

# Menampilkan visualisasi berdasarkan pilihan
if visualization_type == "Distribusi Kategorikal":
    # Menentukan jumlah subplot per baris
    num_cols = 2
    num_rows = (len(categoricals) + num_cols - 1) // num_cols

    # Membuat figure dan axes untuk subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, num_rows * 10))
    axes = axes.flatten()

    # Menampilkan count plot untuk setiap kolom kategorikal
    for idx, col in enumerate(categoricals):
        ax = sns.countplot(x=col, data=df, palette=['blue', 'orange'], ax=axes[idx])
        ax.set_title(f'Count Plot of {col}', fontsize=24)
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel('Count', fontsize=20)

        # Anotasi jumlah di atas setiap bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 8), textcoords='offset points', fontsize=16)

    # Menghapus axes kosong jika ada
    for j in range(len(categoricals), len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    st.pyplot(fig)  # Menampilkan plot di Streamlit
    st.write("Untuk kolom kategori Education dan Self-employed, distribusinya hampir seimbang antara dua kategori, sedangkan untuk Loan Status, jumlah yang Approved jauh lebih banyak dibandingkan yang Rejected.")
    
elif visualization_type == "Distribusi Numerik":
    # Menyesuaikan ukuran figure agar memuat semua plot
    plt.figure(figsize=(20, 15))

    # Menghitung jumlah baris dan kolom secara otomatis
    num_plots = len(numericals)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    # Plotting univariat untuk kolom numerik
    for i in range(num_plots):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[numericals[i]], kde=True, color='blue')
        plt.title(f'Distribution of {numericals[i]}', fontsize=20)
        plt.xlabel(numericals[i], fontsize=16)
        plt.ylabel('Frequency', fontsize=16)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    st.pyplot(plt)  # Menampilkan plot di Streamlit
    st.write("Distribusi data menunjukkan bahwa variabel jumlah tanggungan/No of dependents dan jangka waktu pinjaman/loan term memiliki pola yang bervariasi, sementara pendapatan tahunan, skor CIBIL, dan aset mewah cenderung mendekati distribusi normal; sebaliknya, jumlah pinjaman serta nilai aset residensial, komersial, dan bank memiliki distribusi yang skewed ke kanan, menandakan bahwa sebagian besar populasi memiliki nilai yang lebih kecil untuk pinjaman dan aset, dengan sedikit yang memiliki jumlah yang jauh lebih besar.")


# Cek Data Outlier
st.write("**d. Cek Data Outlier:**")

# Menampilkan ringkasan statistik
st.write("**Ringkasan Statistik DataFrame:**")
st.write(df.describe())


# Menampilkan Boxplot
st.write("**Boxplot Cek Data Outlier:**")

# Adjust the figure size for better readability
plt.figure(figsize=(20, 6))  # Ukuran diperbesar untuk tampilan yang lebih baik

# List of features to plot
features = ['residential_assets_value', 'commercial_assets_value', 'bank_asset_value']

# Loop through the features and plot
for i in range(len(features)):
    plt.subplot(1, 3, i + 1)  # Set layout to 1 row and 3 columns
    
    # Set color to red for these features
    sns.boxplot(y=df[features[i]], color='red')
    plt.title(features[i], fontsize=20)  # Meningkatkan ukuran font judul
    plt.xlabel(features[i], fontsize=16)  # Meningkatkan ukuran font label sumbu x
    plt.ylabel('Value', fontsize=16)  # Meningkatkan ukuran font label sumbu y

plt.tight_layout()  # Adjusts subplot layout to prevent overlap

# Display the box plot in Streamlit
st.pyplot(plt)  # Menampilkan plot di Streamlit

# Resetting the figure to clear the current plot
plt.clf()  # Clear the current figure for future plots

st.write("Boxplot menunjukkan outlier signifikan pada residential_assets_value, commercial_assets_value, dan bank_asset_value, sementara variabel lainnya memiliki distribusi lebih seimbang tanpa outlier mencolok. Pengecekan dengan df.describe mengungkapkan nilai negatif pada residential_assets_value, yang kemungkinan merupakan kesalahan pencatatan. Nilai outlier tinggi tidak perlu dihapus karena mungkin mencerminkan nilai aset yang tinggi secara wajar, namun nilai negatif perlu ditinjau lebih lanjut")

# Data Preprocessing
st.subheader("D. Data Prepocessing")
st.write("**a. Data Cleansing**")

# 1. Cek data missing value
missing_values = df.isnull().sum()
st.write("**1. Missing Values:**")
st.write(missing_values[missing_values > 0])  # Hanya menampilkan kolom dengan missing values
st.write("Tidak terdapat missing value")

# 2. Cek data duplikat
duplicates = df.duplicated().sum()
st.write("**2. Duplicate Values:**", duplicates)
st.write("Tidak terdapat data duplikat")

# 3. Cek outlier
# Checking for negative values in numerical columns
numeric_columns = df.select_dtypes(include=['number']).columns
anomalies = {col: df[df[col] < 0][col].count() for col in numeric_columns}
st.write("**3. Outlier (Negative Values):**")
st.write(anomalies)

# Menghapus baris yang mengandung nilai negatif pada kolom numerik dan memperbarui df
df = df[(df[numeric_columns] >= 0).all(axis=1)]

# Menampilkan DataFrame yang telah diperbarui
st.write("**Updated Data after Removing Negative Values:**")
st.write(df.describe())

st.write("**b. Membuat Feature Baru**")
st.write("**1. Membuat 5 Feature Baru:**")
# Menampilkan penjelasan fitur baru di Streamlit
# Menampilkan penjelasan fitur baru di Streamlit dengan ukuran huruf lebih kecil
st.markdown("<h4 style='font-size: 14px;'>Penjelasan Fitur Baru</h4>", unsafe_allow_html=True)

st.write("**cibil_category**: Kategori ini mengklasifikasikan skor CIBIL peminjam menjadi lima tingkat berdasarkan rentang nilai skor CIBIL, yaitu:")
st.write("   - **Very Poor**: skor kurang dari 550")
st.write("   - **Poor**: skor antara 550 hingga 649")
st.write("   - **Fair**: skor antara 650 hingga 699")
st.write("   - **Good**: skor antara 700 hingga 749")
st.write("   - **Excellent**: skor 750 ke atas.")
st.write("   Kategori ini memberikan gambaran tentang kesehatan kredit peminjam, di mana skor yang lebih tinggi menunjukkan kredibilitas yang lebih baik dalam hal pengelolaan utang.")

st.write("**total_assets_value**: Fitur ini menghitung total nilai aset peminjam dengan menjumlahkan nilai dari semua jenis aset yang dimiliki, termasuk aset residensial, komersial, mewah, dan nilai aset bank.")

st.write("**debt_to_income_ratio**: Rasio ini menggambarkan proporsi total utang yang diminta (loan_amount) terhadap pendapatan tahunan peminjam (income_annum), yang menunjukkan beban utang relatif terhadap kemampuan mereka untuk membayar.")

st.write("**loan_to_value_ratio**: Rasio ini mengukur proporsi jumlah pinjaman (loan_amount) terhadap total nilai aset peminjam (total_assets_value), memberikan gambaran tentang risiko yang diambil oleh pemberi pinjaman terhadap nilai aset yang dijaminkan.")

st.write("**dependents_to_income_ratio**: Rasio ini menunjukkan proporsi jumlah tanggungan (no_of_dependents) peminjam terhadap pendapatan tahunan mereka (income_annum), yang dapat memberikan wawasan tentang kemampuan peminjam untuk mengelola tanggungan dalam hubungannya dengan penghasilan mereka.")

# Membuat salinan df sebagai df1
df1 = df.copy()

# Fungsi untuk menghitung kategori CIBIL score
def categorize_cibil(score):
    if score < 550:
        return 'Very Poor'
    elif score < 650:
        return 'Poor'
    elif score < 700:
        return 'Fair'
    elif score < 750:
        return 'Good'
    else:
        return 'Excellent'

# Menambahkan kolom cibil_category
df1['cibil_category'] = df1['cibil_score'].apply(categorize_cibil)

# Membuat fitur baru
# 1. Total nilai aset
df1['total_assets_value'] = (df1['residential_assets_value'] +
                             df1['commercial_assets_value'] +
                             df1['luxury_assets_value'] +
                             df1['bank_asset_value'])

# 3. Debt to Income Ratio
df1['debt_to_income_ratio'] = df1['loan_amount'] / df1['income_annum']

# 4. Menghitung Loan to Value Ratio
df1['loan_to_value_ratio'] = df1['loan_amount'] / df1['total_assets_value']

# Rasio tanggungan
df1['dependents_to_income_ratio'] = df1['no_of_dependents'] / df1['income_annum']

# Menghapus kolom yang sudah tidak diperlukan
df1.drop(['residential_assets_value', 'commercial_assets_value', 
         'luxury_assets_value', 'bank_asset_value'], axis=1, inplace=True)

# Menampilkan DataFrame df1 di Streamlit
st.write("### DataFrame dengan fitur baru:")
st.dataframe(df1)

# Deep Dive EDA
st.write("**c. Deep Dive EDA**")

# Pilihan untuk hubungan yang ingin ditampilkan
option = st.selectbox(
    "Pilih Grafik Korelasi terhadap Loan Status yang ingin ditampilkan:",
    ("CIBIL Score", "CIBIL Category", "Education", "Loan Term", 
     "Self Employed", "No of Dependents", "Debt to Income Ratio", "Loan to Value Ratio")
)

# Fungsi untuk menetapkan warna berdasarkan status pinjaman
def set_color(status):
    return 'blue' if status == 'Approved' else 'orange'

if option == "CIBIL Score":
    st.write("**Hubungan antara CIBIL Score dan Loan Status**")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df1, x='loan_status', y='cibil_score', palette={'Approved': 'blue', 'Rejected': 'orange'})
    plt.title('CIBIL Score berdasarkan Status Pinjaman')
    plt.xlabel('Loan Status')
    plt.ylabel('CIBIL Score')
    st.pyplot(plt)
    st.write("Pengajuan pinjaman dengan CIBIL score yang lebih tinggi sekitar 750 cenderung disetujui, sementara pengajuan dengan skor lebih rendah sekitar 550  lebih sering ditolak.")

elif option == "CIBIL Category":
    st.write("**Hubungan antara Kategori CIBIL Category dan Loan Status**")
    frequency = df1.groupby(['cibil_category', 'loan_status']).size().reset_index(name='count')
    plt.figure(figsize=(15, 8))
    bar_plot = sns.barplot(data=frequency, x='cibil_category', y='count', hue='loan_status', palette={'Approved': 'blue', 'Rejected': 'orange'})
    plt.xlabel('CIBIL Category', fontsize=14)
    plt.ylabel('Jumlah', fontsize=14)
    plt.title('Hubungan antara CIBIL Category dan Loan Status', fontsize=16)
    plt.legend(title='Loan Status')
    plt.xticks(rotation=45)

    for p in bar_plot.patches:
        bar_plot.annotate(f'{int(p.get_height())}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', 
                          fontsize=10, color='black', 
                          xytext=(0, 5),  # Jarak antara angka dan batang
                          textcoords='offset points')

    st.pyplot(plt)
    st.write("Rentang kategori CIBIL score menunjukkan bahwa individu dengan skor Very Poor (kurang dari 550) hanya memiliki 1 Approved dan 1591 Rejected, sedangkan skor Excellent (750 ke atas) mencatat 1040 Approved dan 5 Rejected, menunjukkan bahwa semakin tinggi CIBIL score, semakin besar kemungkinan untuk mendapatkan pinjaman yang disetujui")


elif option == "Education":
    st.write("Hubungan antara Education dan Loan Status")
    plt.figure(figsize=(10, 5))
    count_plot = sns.countplot(data=df1, x='education', hue='loan_status', palette={'Approved': 'blue', 'Rejected': 'orange'})
    plt.title('Loan Status berdasarkan Education')
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.legend(title='Loan Status')

    for p in count_plot.patches:
        count_plot.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=10, color='black', 
                            xytext=(0, 2),  # Geser angka sedikit ke bawah
                            textcoords='offset points')

    st.pyplot(plt)
    st.write("status pendidikan memiliki pengaruh kecil terhadap keputusan pinjaman, di mana lulusan sekolah sedikit lebih banyak mendapatkan persetujuan dibandingkan non-lulusan, namun perbedaannya tidak terlalu signifikan.")


elif option == "Loan Term":
    st.write("Hubungan antara Loan Term dan Loan Status")
    
    # Menghitung persentase loan status berdasarkan loan term
    loan_counts = df1.groupby(['loan_term', 'loan_status']).size().reset_index(name='count') 
    total_counts = loan_counts.groupby('loan_term')['count'].sum().reset_index(name='total')
    loan_counts = loan_counts.merge(total_counts, on='loan_term')
    loan_counts['percentage'] = (loan_counts['count'] / loan_counts['total']) * 100
    
    # Membuat data pivot untuk visualisasi
    pivot_data = loan_counts.pivot(index='loan_term', columns='loan_status', values='percentage').fillna(0)

    # Mengatur ukuran lebih kecil menggunakan subplots
    fig, ax = plt.subplots(figsize=(7, 4))  # Ukuran lebih kecil dari sebelumnya (10,5)
    
    # Membuat stacked bar plot
    pivot_data.plot(kind='bar', stacked=True, color=['blue', 'orange'], ax=ax)
    ax.set_title('Persentase Loan Status berdasarkan Loan Term')
    ax.set_xlabel('Loan Term')
    ax.set_ylabel('Persentase (%)')
    ax.set_xticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.index, rotation=0)
    
    # Menambahkan legend
    ax.legend(title='Loan Status', labels=['Approved', 'Rejected'], loc='upper right')
    
    # Menampilkan chart di Streamlit
    st.pyplot(fig)
    st.write("Persentase status pinjaman menunjukkan bahwa tingkat persetujuan (Approved) mulai menurun dan penolakan (Rejected) meningkat setelah 4 tahun jangka waktu pinjaman.")


elif option == "Self Employed":
    st.write("Hubungan antara Self Employed dan Loan Status")
    plt.figure(figsize=(10, 5))
    count_plot = sns.countplot(data=df1, x='self_employed', hue='loan_status', palette={'Approved': 'blue', 'Rejected': 'orange'})

    for p in count_plot.patches:
        count_plot.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=10, color='black', 
                            xytext=(0, 2),  # Jarak antara angka dan batang
                            textcoords='offset points')

    plt.title('Loan Status berdasarkan Status Self Employed')
    plt.xlabel('Self Employed')
    plt.ylabel('Count')
    plt.legend(title='Loan Status')
    st.pyplot(plt)
    st.write("Prediksi status pinjaman memiliki peluang persetujuan yang hampir seimbang antara individu yang bekerja sendiri dan yang tidak, dengan sedikit lebih banyak persetujuan pada kelompok yang bekerja sendiri.")


elif option == "No of Dependents":
    st.write("Hubungan antara Number of Dependents dan Loan Status")
    dependents_counts = df1.groupby(['no_of_dependents', 'loan_status']).size().reset_index(name='count')
    plt.figure(figsize=(10, 5))
    count_plot = sns.barplot(data=dependents_counts, x='no_of_dependents', y='count', hue='loan_status', palette={'Approved': 'blue', 'Rejected': 'orange'})

    for p in count_plot.patches:
        count_plot.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=10, color='black', 
                            xytext=(0, 3),  # Jarak antara angka dan batang
                            textcoords='offset points')

    plt.title('Jumlah Loan Status berdasarkan No of Dependents/Jumlah Tanggungan')
    plt.xlabel('Jumlah Tanggungan')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=0)
    plt.legend(title='Loan Status', loc='upper right')
    st.pyplot(plt)
    st.write("Prediksi status pinjaman menunjukkan peluang persetujuan yang hampir seimbang antara individu dengan berbagai jumlah tanggungan, karena jumlah yang disetujui dan ditolak hampir sama.")
    
elif option == "Debt to Income Ratio":
    st.write("Hubungan Debt to Income Ratio dan Loan Status")
    median_debt_to_income_ratio = df1.groupby('loan_status')['debt_to_income_ratio'].median().reset_index()
    plt.figure(figsize=(10, 5))
    barplot = sns.barplot(data=median_debt_to_income_ratio, x='loan_status', y='debt_to_income_ratio', palette={'Approved': 'blue', 'Rejected': 'orange'})

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', 
                         fontsize=10, color='black', 
                         xytext=(0, 3),  # Jarak antara angka dan batang
                         textcoords='offset points')

    plt.title('Median Debt-to-Income Ratio berdasarkan Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Median Debt-to-Income Ratio')
    st.pyplot(plt)
    st.write("Median debt-to-income ratio yang lebih tinggi untuk pinjaman yang disetujui, yaitu 3,05 dibandingkan 2,88 untuk yang ditolak, menunjukkan adanya pengaruh terhadap keputusan pinjaman, meskipun pengaruhnya tergolong kecil. Dan Individu dengan debt-to-income ratio yang lebih tinggi cenderung memiliki peluang lebih besar untuk mendapatkan persetujuan pinjaman.")

elif option == "Loan to Value Ratio":
    st.write("Hubungan Loan to Value Ratio dan Loan Status")
    median_loan_to_value_ratio = df1.groupby('loan_status')['loan_to_value_ratio'].median().reset_index()
    plt.figure(figsize=(10, 5))
    barplot = sns.barplot(data=median_loan_to_value_ratio, x='loan_status', y='loan_to_value_ratio', palette={'Approved': 'blue', 'Rejected': 'orange'})

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', 
                         fontsize=10, color='black', 
                         xytext=(0, 3),  # Jarak antara angka dan batang
                         textcoords='offset points')

    plt.title('Median Loan to Value Ratio berdasarkan Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Median Loan to Value Ratio')
    st.pyplot(plt)
    st.write("Median loan-to-value ratio yang hampir sama, yaitu 0,47 untuk yang disetujui dan 0,46 untuk yang ditolak, menunjukkan bahwa rasio ini memiliki pengaruh kecil terhadap keputusan persetujuan pinjaman.")

# Menutup plt untuk menghindari tampilan ganda di Streamlit
plt.close()


# Deep Dive EDA
st.write("**d. Encoding Categorical**")

# Tampilkan informasi tentang encoding
st.write("Menggunakan One Hot Encoding: education, self_employed")
st.write("Encode: cibil_category, loan_status")


# One-Hot Encoding untuk kolom 'education'
df1 = pd.get_dummies(df1, columns=['education'], prefix='education')

# One-Hot Encoding untuk kolom 'self_employed'
df1 = pd.get_dummies(df1, columns=['self_employed'], prefix='self_employed')

# Label Encoding secara manual untuk kolom 'loan_status'
df1['loan_status'] = df1['loan_status'].replace({'Rejected': 0, 'Approved': 1})

# Label Encoding secara manual untuk kolom 'cibil_category'
df1['cibil_category'] = df1['cibil_category'].replace({
    'Very Poor': 0, 
    'Poor': 1, 
    'Fair': 2, 
    'Good': 3, 
    'Excellent': 4
})

# Menampilkan hasil encoding di Streamlit
st.write("**Hasil Encoding Data Frame**")
st.dataframe(df1.head())  # Tampilkan 5 baris pertama dari df1

# Menghapus kolom 'loan_id' dari df1
df1 = df1.drop(columns=['loan_id'])

# Memindahkan kolom 'loan_status' ke posisi terakhir
cols = [col for col in df1.columns if col != 'loan_status']  # Ambil semua kolom kecuali 'loan_status'
df1 = df1[cols + ['loan_status']]  # Menambahkan 'loan_status' di akhir

# Feature Selection 
st.write("**e. Feature Selection**")

# Membuat tab untuk sebelum dan setelah feature selection
tab1, tab2 = st.tabs(["Sebelum Feature Selection", "Setelah Feature Selection"])

with tab1:
    st.write("1. Correlation Heatmap Sebelum Feature Selection")

    # Menghitung matriks korelasi sebelum feature selection
    corr_matrix_before = df1.corr()

    # Mengatur ukuran figure untuk heatmap
    plt.figure(figsize=(12, 8))

    # Membuat heatmap untuk korelasi antar fitur sebelum feature selection
    sns.heatmap(corr_matrix_before, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

    # Menambahkan judul
    plt.title('Correlation Matrix of Features before Feature Selection', fontsize=16)

    # Menampilkan heatmap di Streamlit
    st.pyplot(plt)

with tab2:
    st.write("2. Correlation Heatmap Setelah Feature Selection")
    st.write("Threshold ditetapkan di atas 0.8 atau di bawah -0.8")
    st.write("Feature yang dihapus adalah: education_Not Graduate, self_employed_No, loan_amount, cibil_category, total_assets_value")

    # Kolom yang ingin dihapus
    kolom_hapus = ['education_Not Graduate', 'self_employed_No', 'loan_amount', 'cibil_category', 'total_assets_value']

    # Menghapus kolom dari DataFrame
    df1_drop = df1.drop(columns=kolom_hapus)

    # Menghitung matriks korelasi setelah feature selection
    corr_matrix_after = df1_drop.corr()

    # Mengatur ukuran figure untuk heatmap
    plt.figure(figsize=(12, 8))

    # Membuat heatmap untuk korelasi antar fitur setelah feature selection
    sns.heatmap(corr_matrix_after, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

    # Menambahkan judul
    plt.title('Correlation Matrix of Features after Feature Selection', fontsize=16)

    # Menampilkan heatmap di Streamlit
    st.pyplot(plt)


# Standarization
st.write("**f. Standarization**")
st.write("Menggunakan Robust Scaler untuk feature: no_of_dependents, income_annum, loan_term, debt_to_income_ratio, loan_to_value_ratio")

from sklearn.preprocessing import RobustScaler

# Memisahkan fitur numerik yang perlu di-scaling
features_to_scale = ['no_of_dependents', 'income_annum', 'loan_term',
                     'debt_to_income_ratio', 'loan_to_value_ratio']

# Membuat objek RobustScaler
scaler = RobustScaler()

# Melakukan scaling pada fitur-fitur yang dipilih
df1_drop[features_to_scale] = scaler.fit_transform(df1_drop[features_to_scale])

# Menampilkan DataFrame setelah scaling di Streamlit
st.write("DataFrame df1 setelah scaling:")
st.dataframe(df1_drop.head())  # Menampilkan 5 baris pertama setelah scaling

# Machine Learning
st.subheader("E. Machine Learning")
st.write("**a. Pemilihan Metrik**")

# Membuat Grafik distribusi data Loan Status
# Menghitung jumlah untuk setiap loan_status
loan_status_counts = df1_drop['loan_status'].value_counts()

# Menghitung persentase
loan_status_percentage = loan_status_counts / loan_status_counts.sum() * 100

# Mengatur ukuran grafik yang lebih kecil
plt.figure(figsize=(2, 2))

# Membuat diagram lingkaran tanpa label langsung di chart
plt.pie(
    loan_status_percentage,
    autopct='%1.1f%%',  # Menampilkan persentase
    startangle=90,
    colors=['#1f77b4', '#ff7f0e'],  # Warna biru untuk Approved dan oranye untuk Rejected
    textprops={'fontsize': 8, 'fontweight': 'bold'}  # Ukuran font lebih kecil
)

# Menambahkan legend dengan font lebih kecil, dan diletakkan di kanan atas tanpa menempel pada grafik
plt.legend(
    ['Approved', 'Rejected'], 
    loc='upper right', 
    fontsize=6,  # Ukuran font lebih kecil untuk legend
    bbox_to_anchor=(1.3, 1)  # Menggeser legend ke kanan agar tidak menempel
)

# Menambahkan judul dengan ukuran lebih kecil
plt.title('Distribusi Persentase Loan Status', fontsize=8, fontweight='bold')  # Ukuran font judul lebih kecil

# Menjaga agar lingkaran tetap berbentuk lingkaran
plt.axis('equal')

# Menampilkan grafik di Streamlit
st.pyplot(plt)

st.write("Karena adanya ketidakseimbangan data, tidak memakai metrik Accuracy. Maka digunakan metrik F1-Score (sebagai metrik utama), Precision, Recall, dan AUC-ROC")

st.write("**b. Split Data**")
st.write("Membagi data menjadi 3 yaitu Data Train (60%), Data Validation(40%), Data Test(40%)")

from sklearn.model_selection import train_test_split

# Memisahkan fitur dan target
X = df1_drop.drop(columns=['loan_status'])
y = df1_drop['loan_status']

# Pembagian langsung: train (60%), validation (20%), test (20%) dengan stratifikasi
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)  # 20% valid, 20% test

# Menampilkan bentuk dari masing-masing set di Streamlit
st.write(f'**X_train shape**: {X_train.shape}')
st.write(f'**X_val shape**: {X_val.shape}')
st.write(f'**X_test shape**: {X_test.shape}')

st.write("**c. Modeling**")
st.write("**c.1 Modeling dengan CIBIL Score**")
st.write("**c.1.1 Membuat Model dengan CIBIL Score dan Hasil Evaluasi**")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

# Tab untuk setiap model
model_names = ["Dummy Classifier", "Logistic Regression", "Support Vector Classifier (SVC)", "Random Forest", "XGBoost"]
tabs = st.tabs(model_names)

for tab, model_name in zip(tabs, model_names):
    with tab:
        if model_name == "Dummy Classifier":
            st.write("**1. Dummy Classifier**")
            baseline_model = DummyClassifier(strategy="stratified")
            baseline_model.fit(X_train, y_train)

            y_train_pred_baseline = baseline_model.predict(X_train)
            y_val_pred_baseline = baseline_model.predict(X_val)

            y_train_pred_proba_baseline = baseline_model.predict_proba(X_train)[:, 1]
            y_val_pred_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Baseline Model (DummyClassifier) pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_baseline, zero_division=0))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_baseline)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_baseline)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_baseline, zero_division=0)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_baseline, zero_division=0)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_baseline, zero_division=0)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Baseline Model (DummyClassifier) pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_baseline, zero_division=0))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_baseline)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_baseline)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_baseline, zero_division=0)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_baseline, zero_division=0)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_baseline, zero_division=0)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_baseline)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Dummy Classifier')
            st.pyplot(plt)

        elif model_name == "Logistic Regression":
            st.write("**2. Logistic Regression**")
            logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
            logreg.fit(X_train, y_train)

            y_train_pred_logreg = logreg.predict(X_train)
            y_val_pred_logreg = logreg.predict(X_val)

            y_train_pred_proba_logreg = logreg.predict_proba(X_train)[:, 1]
            y_val_pred_proba_logreg = logreg.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model Logistic Regression pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_logreg))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_logreg)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_logreg)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_logreg)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_logreg)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_logreg)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model Logistic Regression pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_logreg))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_logreg)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_logreg)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_logreg)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_logreg)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_logreg)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_logreg)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Logistic Regression')
            st.pyplot(plt)

        elif model_name == "Support Vector Classifier (SVC)":
            st.write("**3. Support Vector Classifier (SVC)**")
            svc = SVC(probability=True, class_weight='balanced')
            svc.fit(X_train, y_train)

            y_train_pred_svc = svc.predict(X_train)
            y_val_pred_svc = svc.predict(X_val)

            y_train_pred_proba_svc = svc.predict_proba(X_train)[:, 1]
            y_val_pred_proba_svc = svc.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model SVC pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_svc))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_svc)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_svc)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_svc)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_svc)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_svc)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model SVC pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_svc))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_svc)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_svc)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_svc)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_svc)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_svc)))
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_svc)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - SVC')
            st.pyplot(plt)

        elif model_name == "Random Forest":
            st.write("**4. Random Forest**")
            rf = RandomForestClassifier(class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)

            y_train_pred_rf = rf.predict(X_train)
            y_val_pred_rf = rf.predict(X_val)

            y_train_pred_proba_rf = rf.predict_proba(X_train)[:, 1]
            y_val_pred_proba_rf = rf.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model Random Forest pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_rf))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_rf)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_rf)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_rf)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_rf)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_rf)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model Random Forest pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_rf))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_rf)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_rf)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_rf)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_rf)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_rf)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_rf)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Random Forest')
            st.pyplot(plt)

        elif model_name == "XGBoost":
            st.write("**5. XGBoost**")
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

            y_train_pred_xgb = xgb.predict(X_train)
            y_val_pred_xgb = xgb.predict(X_val)

            y_train_pred_proba_xgb = xgb.predict_proba(X_train)[:, 1]
            y_val_pred_proba_xgb = xgb.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model XGBoost pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_xgb))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_xgb)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_xgb)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_xgb)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_xgb)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_xgb)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model XGBoost pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_xgb))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_xgb)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_xgb)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_xgb)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_xgb)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_xgb)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_xgb)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - XGBoost')
            st.pyplot(plt)
            

st.write("**c.1.2 Analisis dan Pemilihan Model Terbaik**")

# Data metrik performa
data_train = {
    'Metric': ['F1-Score', 'Precision', 'Recall', 'AUC-ROC'],
    'Dummy Classifier': [
        f1_score(y_train, y_train_pred_baseline, zero_division=0), 
        precision_score(y_train, y_train_pred_baseline, zero_division=0),
        recall_score(y_train, y_train_pred_baseline, zero_division=0),
        roc_auc_score(y_train, y_train_pred_proba_baseline)
    ],
    'Logistic Regression': [
        f1_score(y_train, y_train_pred_logreg),
        precision_score(y_train, y_train_pred_logreg),
        recall_score(y_train, y_train_pred_logreg),
        roc_auc_score(y_train, y_train_pred_proba_logreg)
    ],
    'SVM': [
        f1_score(y_train, y_train_pred_svc),
        precision_score(y_train, y_train_pred_svc), 
        recall_score(y_train, y_train_pred_svc),
        roc_auc_score(y_train, y_train_pred_proba_svc)
    ],
    'Random Forest': [
        f1_score(y_train, y_train_pred_rf), 
        precision_score(y_train, y_train_pred_rf), 
        recall_score(y_train, y_train_pred_rf),
        roc_auc_score(y_train, y_train_pred_proba_rf)
    ],
    'XGBoost': [
        f1_score(y_train, y_train_pred_xgb), 
        precision_score(y_train, y_train_pred_xgb), 
        recall_score(y_train, y_train_pred_xgb),
        roc_auc_score(y_train, y_train_pred_proba_xgb)
    ]
}

# Membuat DataFrame untuk train
df_metrics = pd.DataFrame(data_train)

# Data metrik performa untuk validasi
data_validasi = {
    'Metric': ['F1-Score', 'Precision', 'Recall','AUC-ROC'],
    'Dummy Classifier': [
        f1_score(y_val, y_val_pred_baseline, zero_division=0), 
        precision_score(y_val, y_val_pred_baseline, zero_division=0),
        recall_score(y_val, y_val_pred_baseline, zero_division=0),
        roc_auc_score(y_val, y_val_pred_proba_baseline)
    ],
    'Logistic Regression': [
        f1_score(y_val, y_val_pred_logreg),
        precision_score(y_val, y_val_pred_logreg),
        recall_score(y_val, y_val_pred_logreg),
        roc_auc_score(y_val, y_val_pred_proba_logreg)
    ],
    'SVM': [
        f1_score(y_val, y_val_pred_svc),
        precision_score(y_val, y_val_pred_svc), 
        recall_score(y_val, y_val_pred_svc),
        roc_auc_score(y_val, y_val_pred_proba_svc)
    ],
    'Random Forest': [
        f1_score(y_val, y_val_pred_rf), 
        precision_score(y_val, y_val_pred_rf), 
        recall_score(y_val, y_val_pred_rf),
        roc_auc_score(y_val, y_val_pred_proba_rf)
    ],
    'XGBoost': [
        f1_score(y_val, y_val_pred_xgb), 
        precision_score(y_val, y_val_pred_xgb), 
        recall_score(y_val, y_val_pred_xgb),
        roc_auc_score(y_val, y_val_pred_proba_xgb)
    ]
}

# Membuat DataFrame untuk validasi
df_metrics1 = pd.DataFrame(data_validasi)

# Menampilkan tabel menggunakan Streamlit
st.write("**Metrik Performa Model**")

# Membuat tab
tabs1, tabs2 = st.tabs(["Train", "Validation"])

# Menampilkan tabel pada tab train
with tabs1:
    st.write("**Metrik Performa untuk Train**")
    st.dataframe(df_metrics)

# Menampilkan tabel pada tab validasi
with tabs2:
    st.write("**Metrik Performa untuk Validation**")
    st.dataframe(df_metrics1)

st.write("Model Random Forest menunjukkan kinerja terbaik di semua metrik, khususnya F1-Score, yang sangat tinggi baik pada data Training maupun Validation, menegaskan kemampuannya dalam memprediksi status pinjaman secara akurat. Maka Model terbaik adalah Random Forest dan akan dilakukan testing pada model tersebut")

st.write("**c.1.3 Testing di Model Terbaik dengan CIBIL Score**")
# Prediksi model pada test set
y_test_pred_rf = rf.predict(X_test)

# Prediksi probabilitas pada test set (untuk AUC-ROC)
y_test_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Tampilkan hasil evaluasi di Streamlit
st.write("**Evaluasi Model Random Forest dengan fitur CIBIL Score pada Data Test**")

st.text("Classification Report:\n" + classification_report(y_test, y_test_pred_rf))
st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_test, y_test_pred_proba_rf)))
st.text("F1-Score: {:.4f}".format(f1_score(y_test, y_test_pred_rf)))
st.text("Precision: {:.4f}".format(precision_score(y_test, y_test_pred_rf)))
st.text("Recall: {:.4f}".format(recall_score(y_test, y_test_pred_rf)))

import streamlit as st

# Menampilkan hasil model Random Forest dalam satu paragraf
st.write("Model **Random Forest** dengan fitur **CIBIL Score** mencapai nilai sempurna di semua metrik (F1-Score, Precision, Recall, ROC AUC) sebesar **1.0** pada data test, menunjukkan kemampuan prediksi yang sangat akurat.")


# Menghitung confusion matrix untuk Random Forest
conf_matrix = confusion_matrix(y_test, y_test_pred_rf)

# Visualisasi confusion matrix untuk Random Forest
st.write("**c.1.4 Confusion Matrix**")
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

# Menambahkan label
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest')
st.pyplot(plt)  # Tampilkan confusion matrix di Streamlit

# Menampilkan nilai confusion matrix dan kesimpulan langsung
st.write("**Hasil Prediksi Loan Approval menggunakan model Random Forest**")
st.write("**True Positives (TP):** 528 pinjaman yang seharusnya disetujui dan benar-benar disetujui.")
st.write("**True Negatives (TN):** 321 pinjaman yang seharusnya ditolak dan benar-benar ditolak.")
st.write("**False Positives (FP):** 0 pinjaman yang salah diprediksi sebagai disetujui.")
st.write("**False Negatives (FN):** 0 pinjaman yang salah diprediksi sebagai ditolak.")
st.write("**Kesimpulan**")
st.write("Model menunjukkan keandalan tinggi dalam memproses aplikasi pinjaman, membantu lembaga keuangan dalam pengambilan keputusan dan manajemen risiko.")


st.write("**c.1.5 Feature Importance dengan CIBIL Score**")

# Menghitung dan menampilkan feature importance
feature_importance = rf.feature_importances_

# Membuat DataFrame untuk menampilkan feature importance
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Mengurutkan fitur berdasarkan importance dari yang tertinggi ke yang terendah
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualisasi feature importance di Streamlit
st.write("**Feature Importance dari Model Random Forest dengan CIBIL Score**")

plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Importance', y='Feature', data=importance_df, color='blue')  # Mengubah warna batang menjadi biru

# Menambahkan label di atas setiap batang dengan 2 angka di belakang koma
for index, value in enumerate(importance_df['Importance']):
    plt.text(value + 0.001, index, f'{value:.2f}', va='center')  # Mengubah menjadi 2 angka di belakang koma

# Menambahkan judul dan label sumbu
plt.title('Feature Importance dari Model Random Forest', fontsize=20)  # Memperbesar ukuran judul
plt.xlabel('Mean Importance', fontsize=16)  # Memperbesar ukuran label sumbu X
plt.ylabel('Feature', fontsize=16)  # Memperbesar ukuran label sumbu Y

# Mengatur ukuran label di sumbu Y dan sumbu X
plt.yticks(fontsize=14)  # Ukuran label sumbu Y
plt.xticks(fontsize=14)  # Ukuran label sumbu X

# Menampilkan grafik dengan layout yang lebih rapi
plt.tight_layout()

# Tampilkan grafik di Streamlit
st.pyplot(plt)

# Menampilkan informasi tentang fitur CIBIL score dan feature importance pada model Random Forest
st.write("**Analisis Feature Importance pada Model Random Forest**")
st.write("Fitur **CIBIL score** dengan importance sebesar **0,84** pada model Random Forest sangat mendominasi pengaruh prediksi, menunjukkan bahwa model sangat bergantung pada fitur ini.")
st.write("Sementara itu, fitur-fitur lain dengan importance sebesar **0,07** atau lebih rendah memiliki kontribusi yang jauh lebih kecil terhadap hasil prediksi.")


st.write("**c.2 Modeling Tanpa CIBIL Score**")

# Load dataset
data = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9i0KlgP1WY52MFh55WxF1bAeoXWUu4a5z3OV-DplxlRRyIEvZxLa58UbUox1W6nlzYEYZw9AF94z-/pub?output=csv')

# Membersihkan nama kolom jika ada spasi ekstra
data.columns = data.columns.str.strip()

# Memastikan tidak ada spasi ekstra di nilai loan_status
data['loan_status'] = data['loan_status'].str.strip()

# Memastikan tidak ada spasi ekstra di nilai education dan self_employed
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()

# Menghapus baris yang mengandung nilai negatif pada kolom numerik dan memperbarui df
data = data[(data[numeric_columns] >= 0).all(axis=1)]

# Membuat fitur baru
# 1. Total nilai aset
data['total_assets_value'] = (data['residential_assets_value'] +
                              data['commercial_assets_value'] +
                              data['luxury_assets_value'] +
                              data['bank_asset_value'])  # Total nilai aset

# 2. Debt to Income Ratio
# Misalkan kita gunakan loan_amount sebagai total utang yang diminta
data['debt_to_income_ratio'] = data['loan_amount'] / data['income_annum']

# 3. Menghitung Loan to Value Ratio
data['loan_to_value_ratio'] = data['loan_amount'] / data['total_assets_value']  # Menggunakan total_assets_value yang sudah dibuat

# 4. Rasio tanggungan terhadap pendapatan
data['dependents_to_income_ratio'] = data['no_of_dependents'] / data['income_annum']

# Menghapus kolom yang sudah tidak diperlukan
data.drop(['residential_assets_value', 'commercial_assets_value', 
         'luxury_assets_value', 'bank_asset_value'], axis=1, inplace=True)

# Menghapus kolom 'loan_id', 'cibil_score' dari data
data = data.drop(columns=['loan_id', 'cibil_score'])

# One-Hot Encoding untuk kolom 'education'
data = pd.get_dummies(data, columns=['education'], prefix='education')

# One-Hot Encoding untuk kolom 'self_employed'
data = pd.get_dummies(data, columns=['self_employed'], prefix='self_employed')

# Label Encoding secara manual untuk kolom 'loan_status'
data['loan_status'] = data['loan_status'].replace({'Rejected': 0, 'Approved': 1})

# Memindahkan kolom 'loan_status' ke posisi terakhir
cols = [col for col in data.columns if col != 'loan_status']  # Ambil semua kolom kecuali 'loan_status'
data = data[cols + ['loan_status']]  # Menambahkan 'loan_status' di akhir


# Kolom yang ingin dihapus
kolom_hapus = ['education_Not Graduate', 'self_employed_No', 'loan_amount','total_assets_value']

# Menghapus kolom
data.drop(columns=kolom_hapus, inplace=True)

# Memisahkan fitur numerik yang perlu di-scaling
features_to_scale = ['no_of_dependents', 'income_annum', 'loan_term',
                     'debt_to_income_ratio', 'loan_to_value_ratio']

# Membuat objek RobustScaler
scaler = RobustScaler()

# Melakukan scaling pada fitur-fitur yang dipilih
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Memisahkan fitur dan target
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Pembagian langsung: train (60%), validation (20%), test (20%) dengan stratifikasi
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)  # 20% valid, 20% test

# Menampilkan bentuk dari masing-masing set di Streamlit
st.write(f'**X_train shape**: {X_train.shape}')
st.write(f'**X_val shape**: {X_val.shape}')
st.write(f'**X_test shape**: {X_test.shape}')

st.write("**c.2.1 Membuat Model tanpa CIBIL Score dan Hasil Evaluasi**")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

# Tab untuk setiap model
model_names = ["Dummy Classifier", "Logistic Regression", "Support Vector Classifier (SVC)", "Random Forest", "XGBoost"]
tabs = st.tabs(model_names)

for tab, model_name in zip(tabs, model_names):
    with tab:
        if model_name == "Dummy Classifier":
            st.write("**1. Dummy Classifier**")
            baseline_model = DummyClassifier(strategy="stratified")
            baseline_model.fit(X_train, y_train)

            y_train_pred_baseline = baseline_model.predict(X_train)
            y_val_pred_baseline = baseline_model.predict(X_val)

            y_train_pred_proba_baseline = baseline_model.predict_proba(X_train)[:, 1]
            y_val_pred_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Baseline Model (DummyClassifier) pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_baseline, zero_division=0))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_baseline)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_baseline)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_baseline, zero_division=0)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_baseline, zero_division=0)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_baseline, zero_division=0)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Baseline Model (DummyClassifier) pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_baseline, zero_division=0))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_baseline)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_baseline)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_baseline, zero_division=0)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_baseline, zero_division=0)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_baseline, zero_division=0)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_baseline)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Dummy Classifier')
            st.pyplot(plt)

        elif model_name == "Logistic Regression":
            st.write("**2. Logistic Regression**")
            logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
            logreg.fit(X_train, y_train)

            y_train_pred_logreg = logreg.predict(X_train)
            y_val_pred_logreg = logreg.predict(X_val)

            y_train_pred_proba_logreg = logreg.predict_proba(X_train)[:, 1]
            y_val_pred_proba_logreg = logreg.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model Logistic Regression pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_logreg))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_logreg)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_logreg)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_logreg)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_logreg)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_logreg)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model Logistic Regression pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_logreg))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_logreg)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_logreg)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_logreg)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_logreg)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_logreg)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_logreg)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Logistic Regression')
            st.pyplot(plt)

        elif model_name == "Support Vector Classifier (SVC)":
            st.write("**3. Support Vector Classifier (SVC)**")
            svc = SVC(probability=True, class_weight='balanced')
            svc.fit(X_train, y_train)

            y_train_pred_svc = svc.predict(X_train)
            y_val_pred_svc = svc.predict(X_val)

            y_train_pred_proba_svc = svc.predict_proba(X_train)[:, 1]
            y_val_pred_proba_svc = svc.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model SVC pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_svc))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_svc)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_svc)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_svc)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_svc)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_svc)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model SVC pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_svc))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_svc)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_svc)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_svc)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_svc)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_svc)))
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_svc)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - SVC')
            st.pyplot(plt)

        elif model_name == "Random Forest":
            st.write("**4. Random Forest**")
            rf = RandomForestClassifier(class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)

            y_train_pred_rf = rf.predict(X_train)
            y_val_pred_rf = rf.predict(X_val)

            y_train_pred_proba_rf = rf.predict_proba(X_train)[:, 1]
            y_val_pred_proba_rf = rf.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model Random Forest pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_rf))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_rf)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_rf)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_rf)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_rf)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_rf)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model Random Forest pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_rf))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_rf)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_rf)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_rf)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_rf)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_rf)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_rf)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - Random Forest')
            st.pyplot(plt)

        elif model_name == "XGBoost":
            st.write("**5. XGBoost**")
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

            y_train_pred_xgb = xgb.predict(X_train)
            y_val_pred_xgb = xgb.predict(X_val)

            y_train_pred_proba_xgb = xgb.predict_proba(X_train)[:, 1]
            y_val_pred_proba_xgb = xgb.predict_proba(X_val)[:, 1]

            # Evaluasi pada Data Pelatihan
            st.write("**Evaluasi Model XGBoost pada Data Pelatihan**")
            st.text("Classification Report:\n" + classification_report(y_train, y_train_pred_xgb))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_train, y_train_pred_xgb)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_train, y_train_pred_proba_xgb)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_train, y_train_pred_xgb)))
            st.text("Precision: {:.4f}".format(precision_score(y_train, y_train_pred_xgb)))
            st.text("Recall: {:.4f}".format(recall_score(y_train, y_train_pred_xgb)))

            # Evaluasi pada Data Validasi
            st.write("**Evaluasi Model XGBoost pada Data Validasi**")
            st.text("Classification Report:\n" + classification_report(y_val, y_val_pred_xgb))
            st.text("Akurasi: {:.4f}".format(accuracy_score(y_val, y_val_pred_xgb)))
            st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_val, y_val_pred_proba_xgb)))
            st.text("F1-Score: {:.4f}".format(f1_score(y_val, y_val_pred_xgb)))
            st.text("Precision: {:.4f}".format(precision_score(y_val, y_val_pred_xgb)))
            st.text("Recall: {:.4f}".format(recall_score(y_val, y_val_pred_xgb)))

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_val, y_val_pred_xgb)
            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix - XGBoost')
            st.pyplot(plt)


st.write("**c.2.2 Analisis dan Pemilihan Model Terbaik Tanpa CIBIL Score**")

# Data metrik performa
data_train = {
    'Metric': ['F1-Score', 'Precision', 'Recall', 'AUC-ROC'],
    'Dummy Classifier': [
        f1_score(y_train, y_train_pred_baseline, zero_division=0), 
        precision_score(y_train, y_train_pred_baseline, zero_division=0),
        recall_score(y_train, y_train_pred_baseline, zero_division=0),
        roc_auc_score(y_train, y_train_pred_proba_baseline)
    ],
    'Logistic Regression': [
        f1_score(y_train, y_train_pred_logreg),
        precision_score(y_train, y_train_pred_logreg),
        recall_score(y_train, y_train_pred_logreg),
        roc_auc_score(y_train, y_train_pred_proba_logreg)
    ],
    'SVM': [
        f1_score(y_train, y_train_pred_svc),
        precision_score(y_train, y_train_pred_svc), 
        recall_score(y_train, y_train_pred_svc),
        roc_auc_score(y_train, y_train_pred_proba_svc)
    ],
    'Random Forest': [
        f1_score(y_train, y_train_pred_rf), 
        precision_score(y_train, y_train_pred_rf), 
        recall_score(y_train, y_train_pred_rf),
        roc_auc_score(y_train, y_train_pred_proba_rf)
    ],
    'XGBoost': [
        f1_score(y_train, y_train_pred_xgb), 
        precision_score(y_train, y_train_pred_xgb), 
        recall_score(y_train, y_train_pred_xgb),
        roc_auc_score(y_train, y_train_pred_proba_xgb)
    ]
}

# Membuat DataFrame untuk train
df_metrics = pd.DataFrame(data_train)

# Data metrik performa untuk validasi
data_validasi = {
    'Metric': ['F1-Score', 'Precision', 'Recall','AUC-ROC'],
    'Dummy Classifier': [
        f1_score(y_val, y_val_pred_baseline, zero_division=0), 
        precision_score(y_val, y_val_pred_baseline, zero_division=0),
        recall_score(y_val, y_val_pred_baseline, zero_division=0),
        roc_auc_score(y_val, y_val_pred_proba_baseline)
    ],
    'Logistic Regression': [
        f1_score(y_val, y_val_pred_logreg),
        precision_score(y_val, y_val_pred_logreg),
        recall_score(y_val, y_val_pred_logreg),
        roc_auc_score(y_val, y_val_pred_proba_logreg)
    ],
    'SVM': [
        f1_score(y_val, y_val_pred_svc),
        precision_score(y_val, y_val_pred_svc), 
        recall_score(y_val, y_val_pred_svc),
        roc_auc_score(y_val, y_val_pred_proba_svc)
    ],
    'Random Forest': [
        f1_score(y_val, y_val_pred_rf), 
        precision_score(y_val, y_val_pred_rf), 
        recall_score(y_val, y_val_pred_rf),
        roc_auc_score(y_val, y_val_pred_proba_rf)
    ],
    'XGBoost': [
        f1_score(y_val, y_val_pred_xgb), 
        precision_score(y_val, y_val_pred_xgb), 
        recall_score(y_val, y_val_pred_xgb),
        roc_auc_score(y_val, y_val_pred_proba_xgb)
    ]
}

# Membuat DataFrame untuk validasi
df_metrics1 = pd.DataFrame(data_validasi)

# Menampilkan tabel menggunakan Streamlit
st.write("**Metrik Performa Model**")

# Membuat tab
tabs1, tabs2 = st.tabs(["Train", "Validation"])

# Menampilkan tabel pada tab train
with tabs1:
    st.write("**Metrik Performa untuk Train**")
    st.dataframe(df_metrics)

# Menampilkan tabel pada tab validasi
with tabs2:
    st.write("**Metrik Performa untuk Validation**")
    st.dataframe(df_metrics1)

st.write("Model Random Forest menunjukkan kinerja terbaik di semua metrik dibandingkan dengan model lain, namun kinerja di data validation mengalami penurunan dibandingakan data train")

st.write("**c.2.3 Testing di Model Terbaik Tanpa CIBIL Score**")
# Prediksi model pada test set
y_test_pred_rf = rf.predict(X_test)

# Prediksi probabilitas pada test set (untuk AUC-ROC)
y_test_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Tampilkan hasil evaluasi di Streamlit
st.write("**Evaluasi Model Random Forest pada Data Test Tanpa CIBIL Score**")

st.text("Classification Report:\n" + classification_report(y_test, y_test_pred_rf))
st.text("Akurasi: {:.4f}".format(accuracy_score(y_test, y_test_pred_rf)))
st.text("AUC-ROC: {:.4f}".format(roc_auc_score(y_test, y_test_pred_proba_rf)))
st.text("F1-Score: {:.4f}".format(f1_score(y_test, y_test_pred_rf)))
st.text("Precision: {:.4f}".format(precision_score(y_test, y_test_pred_rf)))
st.text("Recall: {:.4f}".format(recall_score(y_test, y_test_pred_rf)))

import streamlit as st

# Menampilkan hasil model Random Forest dalam satu paragraf
st.write("Model Random Forest tanpa fitur CIBIL Score menunjukkan performa yang cukup baik, namun belum sempurna. Dengan nilai AUC-ROC sebesar 0.6065, F1-Score 0.7145, Precision 0.6448, dan Recall 0.8011, model ini masih memiliki keterbatasan dalam membedakan kelas positif dan negatif, meskipun recall-nya sudah cukup tinggi. Optimalisasi lebih lanjut diperlukan untuk meningkatkan akurasi keseluruhan.")


# Menghitung confusion matrix untuk Random Forest
conf_matrix = confusion_matrix(y_test, y_test_pred_rf)

# Visualisasi confusion matrix untuk Random Forest
st.write("**c.2.4 Confusion Matrix Tanpa CIBIL Score**")
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

# Menambahkan label
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest')
st.pyplot(plt)  # Tampilkan confusion matrix di Streamlit

# Menampilkan hasil confusion matrix dan kesimpulan di Streamlit
st.write("**Hasil Prediksi Loan Approval menggunakan model Random Forest**")
st.write("**True Positives (TP):** 423 pinjaman yang seharusnya disetujui dan benar-benar disetujui.")
st.write("**True Negatives (TN):** 88 pinjaman yang seharusnya ditolak dan benar-benar ditolak.")
st.write("**False Positives (FP):** 233 pinjaman yang salah diprediksi sebagai disetujui.")
st.write("**False Negatives (FN):** 105 pinjaman yang salah diprediksi sebagai ditolak.")

st.write("**Kesimpulan**")
st.write("Model Random Forest menunjukkan bahwa meskipun ada sejumlah pinjaman yang salah diprediksi sebagai disetujui (FP), jumlah pinjaman yang benar-benar disetujui (TP) cukup tinggi.")
st.write("Ini mengindikasikan bahwa model memiliki potensi, tetapi juga menunjukkan adanya ruang untuk perbaikan dalam mengurangi kesalahan prediksi.")


st.write("**c.2.5 Feature Importance Tanpa CIBIL Score**")

# Menghitung dan menampilkan feature importance
feature_importance = rf.feature_importances_

# Membuat DataFrame untuk menampilkan feature importance
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Mengurutkan fitur berdasarkan importance dari yang tertinggi ke yang terendah
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualisasi feature importance di Streamlit
st.write("**Feature Importance dari Model Random Forest Tanpa CIBIL Score**")

plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Importance', y='Feature', data=importance_df, color='blue')  # Mengubah warna batang menjadi biru

# Menambahkan label di atas setiap batang dengan 2 angka di belakang koma
for index, value in enumerate(importance_df['Importance']):
    plt.text(value + 0.001, index, f'{value:.2f}', va='center')  # Mengubah menjadi 2 angka di belakang koma

# Menambahkan judul dan label sumbu
plt.title('Feature Importance dari Model Random Forest', fontsize=20)  # Memperbesar ukuran judul
plt.xlabel('Mean Importance', fontsize=16)  # Memperbesar ukuran label sumbu X
plt.ylabel('Feature', fontsize=16)  # Memperbesar ukuran label sumbu Y

# Mengatur ukuran label di sumbu Y dan sumbu X
plt.yticks(fontsize=14)  # Ukuran label sumbu Y
plt.xticks(fontsize=14)  # Ukuran label sumbu X

# Menampilkan grafik dengan layout yang lebih rapi
plt.tight_layout()

# Tampilkan grafik di Streamlit
st.pyplot(plt)

# Menampilkan informasi tentang feature importance tanpa CIBIL score
st.write("**Analisis Feature Importance Tanpa CIBIL Score**")
st.write("Tanpa **CIBIL score**, fitur **debt to income ratio** dengan importance sebesar **0,24** menjadi fitur yang paling berpengaruh terhadap prediksi model.")
st.write("Diikuti oleh **loan to value ratio** (importance: **0,22**) dan **income annum** (importance: **0,19**).")

st.subheader("F. Conclusion & Recommendation")

# Menampilkan kesimpulan analisis di Streamlit
st.write("**a.Conclusion**")

# CIBIL Score
st.write("**1. CIBIL Score**")
st.write("**CIBIL score** memegang peranan penting dalam keputusan persetujuan pinjaman, di mana peminjam dengan skor tinggi memiliki peluang lebih besar untuk disetujui.")
st.write("Peminjam dengan skor **'Very Poor' (kurang dari 550)** memiliki tingkat penolakan yang sangat tinggi.")

# Loan Term
st.write("**2. Loan Term**")
st.write("**Jangka waktu pinjaman** berpengaruh terhadap keputusan, dengan penurunan tingkat persetujuan yang terlihat setelah **4 tahun**.")

# Debt-to-Income Ratio dan Loan-to-Value Ratio
st.write("**3. Debt-to-Income Ratio dan Loan-to-Value Ratio**")
st.write("Kedua rasio ini menunjukkan pengaruh terhadap prediksi model, tetapi tidak sekuat **CIBIL score**.")

# Feature Importance
st.write("**4.Feature Importance**")
st.write("- **Model Dengan CIBIL score**, model menunjukkan ketergantungan tinggi pada fitur CIBIL Score, yang memiliki **importance tertinggi**.")
st.write("- **Model Tanpa CIBIL score**, **Debt to Income Ratio** menjadi fitur terpenting, diikuti oleh **Loan to Value Ratio** dan **Income Annum**.")


# Menampilkan rekomendasi di Streamlit
st.write("**b.Recomendation**")

# Rekomendasi CIBIL Score
st.write("**1. Fokus pada CIBIL Score**")
st.write("Gunakan **CIBIL score** sebagai indikator utama dalam penilaian pinjaman untuk meningkatkan akurasi keputusan.")

# Analisis Pinjaman Jangka Panjang
st.write("**2. Analisis Pinjaman Jangka Panjang**")
st.write("Tinjau risiko pinjaman dengan jangka waktu lebih dari **4 tahun** untuk memahami penurunan tingkat persetujuan dan mengambil langkah mitigasi yang efektif.")

# Pengembangan Model Tanpa CIBIL Score
st.write("**3.Pengembangan Model Tanpa Fitur CIBIL Score**")
st.write("Pertimbangkan untuk menambahkan fitur lain seperti **riwayat pembayaran pinjaman sebelumnya** untuk menggantikan **CIBIL score** dalam skenario tertentu.")













