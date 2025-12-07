import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

st.title("Предсказание цены автомобиля")

@st.cache_resource
def load_model():
    with open('model_artifacts.pickle', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_eda_data():
    with open('eda_data.pickle', 'rb') as f:
        return pickle.load(f)

try:
    model_artifacts = load_model()
    eda_data = load_eda_data()
except FileNotFoundError:
    st.error("Файлы модели не найдены! Сначала запустите ноутбук для создания pickle файлов.")
    st.stop()


page = st.sidebar.selectbox(
    "Выберите раздел",
    ["EDA и визуализации", "Предсказание цены", "Веса модели"]
)

if page == "EDA и визуализации":
    st.header("Исследовательский анализ данных (EDA)")

    df_train = eda_data['df_train']
    correlation_matrix = eda_data['correlation_matrix']

    st.subheader("Основная статистика")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Записей в train", df_train.shape[0])
    with col2:
        st.metric("Признаков", df_train.shape[1])
    with col3:
        st.metric("Средняя цена", f"{df_train['selling_price'].mean():,.0f}")
    with col4:
        st.metric("Медианная цена", f"{df_train['selling_price'].median():,.0f}")

   
    viz_type = st.selectbox(
        "Выберите тип визуализации",
        ["Распределение цен", "Корреляционная матрица", "Цена по году выпуска",
         "Распределение по типу топлива", "Распределение по трансмиссии",
         "Зависимость цены от пробега", "Распределение мощности двигателя"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if viz_type == "Распределение цен":
        sns.histplot(df_train['selling_price'], bins=50, kde=True, ax=ax)
        ax.set_xlabel('Цена продажи')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение цен автомобилей')
        ax.set_xlim(0, 2000000)

    elif viz_type == "Корреляционная матрица":
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Корреляционная матрица числовых признаков')

    elif viz_type == "Цена по году выпуска":
        sns.scatterplot(data=df_train, x='year', y='selling_price', alpha=0.5, ax=ax)
        ax.set_xlabel('Год выпуска')
        ax.set_ylabel('Цена продажи')
        ax.set_title('Зависимость цены от года выпуска')
        ax.set_ylim(0, 3000000)

    elif viz_type == "Распределение по типу топлива":
        sns.boxplot(data=df_train, x='fuel', y='selling_price', palette='Set2', ax=ax)
        ax.set_xlabel('Тип топлива')
        ax.set_ylabel('Цена продажи')
        ax.set_title('Распределение цены по типу топлива')
        ax.set_ylim(0, 2000000)

    elif viz_type == "Распределение по трансмиссии":
        sns.boxplot(data=df_train, x='transmission', y='selling_price', palette='Set3', ax=ax)
        ax.set_xlabel('Тип трансмиссии')
        ax.set_ylabel('Цена продажи')
        ax.set_title('Распределение цены по типу трансмиссии')
        ax.set_ylim(0, 2000000)

    elif viz_type == "Зависимость цены от пробега":
        sns.scatterplot(data=df_train, x='km_driven', y='selling_price', alpha=0.5, ax=ax)
        ax.set_xlabel('Пробег (км)')
        ax.set_ylabel('Цена продажи')
        ax.set_title('Зависимость цены от пробега')
        ax.set_xlim(0, 500000)
        ax.set_ylim(0, 2000000)

    elif viz_type == "Распределение мощности двигателя":
        sns.histplot(df_train['max_power'], bins=30, kde=True, ax=ax)
        ax.set_xlabel('Мощность (bhp)')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение мощности двигателей')

    st.pyplot(fig)
    plt.close()

    st.subheader("Описательная статистика числовых признаков")
    st.dataframe(df_train.describe())

elif page == "Предсказание цены":
    st.header("Предсказание цены автомобиля")

    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    encoder = model_artifacts['encoder']
    numerical_features = model_artifacts['numerical_features']
    categorical_features = model_artifacts['categorical_features']

    input_method = st.radio(
        "Выберите способ ввода данных",
        ["Ручной ввод", "Загрузка CSV файла"]
    )

    if input_method == "Ручной ввод":
        st.subheader("Введите характеристики автомобиля")

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2015)
            km_driven = st.number_input("Пробег (км)", min_value=0, max_value=1000000, value=50000)
            mileage = st.number_input("Расход топлива (kmpl)", min_value=0.0, max_value=50.0, value=18.0)
            engine = st.number_input("Объём двигателя (CC)", min_value=500, max_value=5000, value=1200)

        with col2:
            max_power = st.number_input("Мощность (bhp)", min_value=20.0, max_value=500.0, value=80.0)
            fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 14])

        if st.button("Предсказать цену", type="primary"):
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'seats': [seats]
            })

            X_numerical = input_data[numerical_features]
            X_categorical = input_data[categorical_features]

            X_encoded = encoder.transform(X_categorical)
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

            X_final = pd.concat([X_numerical.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
            X_scaled = scaler.transform(X_final)

            prediction = model.predict(X_scaled)[0]

            st.success(f"### Предсказанная цена: {prediction:,.0f}")

    else:
        st.subheader("Загрузите CSV файл")

        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(df_upload.head())

            if st.button("Предсказать цены", type="primary"):
                try:
                    required_cols = numerical_features + categorical_features
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]

                    if missing_cols:
                        st.error(f"Отсутствуют столбцы: {missing_cols}")
                    else:
                        X_numerical = df_upload[numerical_features]
                        X_categorical = df_upload[categorical_features]

                        X_encoded = encoder.transform(X_categorical)
                        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

                        X_final = pd.concat([X_numerical.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
                        X_scaled = scaler.transform(X_final)

                        predictions = model.predict(X_scaled)

                        df_results = df_upload.copy()
                        df_results['predicted_price'] = predictions

                        st.success("Предсказания выполнены!")
                        st.dataframe(df_results)

                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="Скачать результаты CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Ошибка при обработке: {str(e)}")

elif page == "Веса модели":

    coefficients = model_artifacts['model_coefficients']
    features = coefficients['features']
    coefs = coefficients['coefficients']


    st.subheader("Информация о модели")
    metrics = model_artifacts['metrics']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² (Train)", f"{metrics['r2_train']:.4f}")
    with col2:
        st.metric("R² (Test)", f"{metrics['r2_test']:.4f}")
    with col3:
        st.metric("Business Metric (Test)", f"{metrics['business_metric_test']:.2%}")

    st.subheader("Коэффициенты модели")

    coef_df = pd.DataFrame({
        'Признак': features,
        'Коэффициент': coefs,
        'Абс. значение': np.abs(coefs)
    }).sort_values('Абс. значение', ascending=False)

    st.dataframe(coef_df)

    st.subheader("Визуализация важности признаков")

    fig, ax = plt.subplots(figsize=(12, 8))

    coef_df_sorted = coef_df.sort_values('Коэффициент')
    colors = ['green' if x > 0 else 'red' for x in coef_df_sorted['Коэффициент']]

    bars = ax.barh(coef_df_sorted['Признак'], coef_df_sorted['Коэффициент'], color=colors)
    ax.set_xlabel('Коэффициент')
    ax.set_ylabel('Признак')
    ax.set_title('Веса Ridge-регрессии')
    ax.axvline(x=0, color='black', linewidth=0.5)

    for bar, val in zip(bars, coef_df_sorted['Коэффициент']):
        ax.text(val, bar.get_y() + bar.get_height()/2,
               f'{val:,.0f}', va='center', ha='left' if val > 0 else 'right', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Интерпретация весов")
    st.markdown("""
    **Выводы по модели:**
    1. **max_power** - самый важный признак. Чем мощнее автомобиль, тем выше цена.
    2. **year** - год выпуска. Новые автомобили стоят дороже.
    3. **engine** - объём двигателя положительно влияет на цену.
    4. **km_driven** - пробег отрицательно влияет на цену (чем больше пробег, тем дешевле).
    """)
