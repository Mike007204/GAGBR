import streamlit as st
import shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# --------- Streamlit caching: support old versions ----------
def _cache_data_decorator():
    # Streamlit >= 1.18
    if hasattr(st, "cache_data"):
        return st.cache_data
    # Fallback: older versions
    if hasattr(st, "cache"):
        return st.cache
    # No caching available
    def identity(func): return func
    return identity

cache_data = _cache_data_decorator()

def main():
    st.title('GA-GBR for Property Valuation🏆🚀')

    st.subheader('Demonstration:')
    st.subheader('1) Raw dataset')
    st.subheader('2) Model hyperparameters and Evalution Metrics')
    st.subheader('3) SHAP: Feature importance')
    st.subheader('4) Make a Prediction')
    st.write('---')

    st.sidebar.title('GA-GBR prototype🏆🚀')
    st.subheader('⭐️ Housing Price dataset')

    @cache_data()
    def load_data():
        # Ensure dataset.csv is in the same folder as this app
        github_url = "https://raw.githubusercontent.com/Mike007204/GAGBR/main/dataset.csv"
        data = pd.read_csv(github_url )
        return data

    @cache_data()
    def preprocess_and_split(df):
        # Separate features and target
        if 'totalPrice' not in df.columns:
            raise ValueError("Dataset must contain a 'totalPrice' column as target.")
        y = df['totalPrice']
        X = df.drop(columns=['totalPrice'])

        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        # Preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop'
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=550
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Build processed feature names for SHAP plots
        feature_names_processed = list(numerical_features)
        if len(categorical_features) > 0:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            # scikit-learn 1.0+ has get_feature_names_out
            if hasattr(cat_encoder, "get_feature_names_out"):
                cat_names = cat_encoder.get_feature_names_out(categorical_features)
            else:
                # Older sklearn fallback
                cat_names = cat_encoder.get_feature_names(categorical_features)
            feature_names_processed.extend(list(cat_names))

        return (X_train_processed, X_test_processed, y_train, y_test,
                preprocessor, numerical_features, categorical_features,
                X.columns.tolist(), feature_names_processed)

    # Load data
    df = load_data()

    # Show raw data
    if st.sidebar.checkbox('Show Raw Data', True):
        st.write(df)
        st.write(f"Dataset shape: {df.shape}")

    # Preprocess and split
    (X_train, X_test, y_train, y_test, preprocessor,
     numerical_features, categorical_features,
     original_features, feature_names_processed) = preprocess_and_split(df)

    st.sidebar.subheader('Model Hyperparameters')

    loss_gb = st.sidebar.selectbox(
        'Loss function',
        options=['huber', 'squared_error', 'absolute_error'],
    )
    n_estimators_gb = st.sidebar.number_input(
        'Number of boosting trees', 50, 500, value=100, step=50, key='n_estimators_gb'
    )
    learning_rate = st.sidebar.slider('Learning rate', 0.01, 0.3, 0.1, 0.01, key='learning_rate')
    max_depth_gb = st.sidebar.number_input(
        'Maximum depth of trees', 1, 10, step=1, value=3, key='max_depth_gb'
    )

    min_samples_leaf = st.sidebar.number_input(
        'Minimum samples in leaf', 1, 10, step=1, value=1, key='min_samples_leaf'
    )

    metrics = st.sidebar.multiselect(
        'What metrics to display?', ('MAE', 'MSE', 'RMSE', 'R2', 'MAPE'),
        default=['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']
    )

    if st.sidebar.button('Train and Show Metrics', key='classify'):
        st.header('🎯GA-GBR Evaluation')

        model_params = {
            'loss': loss_gb,
            'n_estimators': int(n_estimators_gb),
            'learning_rate': float(learning_rate),
            'max_depth': int(max_depth_gb),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': 0.3,
            'random_state': 21
        }

        model = GradientBoostingRegressor(**model_params)

        with st.spinner('Training model...'):
            model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        model_data = {
            'pipeline': full_pipeline,
            'preprocessor': preprocessor,
            'model': model,
            'original_features': original_features,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'feature_names_processed': feature_names_processed
        }

        filename = 'trained_model_pipeline.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        st.success('Model pipeline trained and saved successfully!')

        st.subheader('Performance Metrics:')
        col1, col2 = st.columns(2)

        with col1:
            st.metric("R² (model.score)", f"{accuracy:.4f}")
            if 'MAE' in metrics:
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            if 'MSE' in metrics:
                st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")

        with col2:
            if 'RMSE' in metrics:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            if 'R2' in metrics:
                st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
            if 'MAPE' in metrics:
                st.metric("MAPE", f"{mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")

        # ---- SHAP on processed features ----
        st.subheader('SHAP Feature Contribution Analysis')
        with st.spinner('Calculating SHAP values...'):
            # TreeExplainer works directly with the trained tree model
            explainer = shap.TreeExplainer(model)
            sample_size = min(100, X_train.shape[0])
            X_sample = X_train[:sample_size]
            shap_values = explainer.shap_values(X_sample)

        # Summary plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names_processed[:X_sample.shape[1]],
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig1)
        plt.clf()
        st.write('---')

        # Bar plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names_processed[:X_sample.shape[1]],
            plot_type="bar",
            show=False
        )
        plt.title('Feature importance based on SHAP values (Bar)')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.clf()

    # -------- Prediction section --------
    st.write('---')
    st.header('Make Predictions')
    st.subheader('Enter Property Features:')

    model_file = 'trained_model_pipeline.pkl'
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        pipeline = model_data.get('pipeline')
        original_features = model_data.get('original_features', [])
        numerical_features = model_data.get('numerical_features', [])
        categorical_features = model_data.get('categorical_features', [])
        model_loaded = True
        st.success(f"Model loaded! Expecting {len(original_features)} features.")
    except FileNotFoundError:
        st.warning("No trained model found. Please train a model first using the sidebar.")
        model_loaded = False
        pipeline = None
        original_features = [col for col in df.columns if col != 'totalPrice']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False
        pipeline = None
        original_features = [col for col in df.columns if col != 'totalPrice']

    st.subheader(f"Input all {len(original_features)} features:")
    feature_descriptions = {
        'Lng': 'Longitude',
        'Lat': 'Latitude',
        'tradeTime': 'Transaction time/date',
        'DOM': 'Days on Market',
        'followers': 'Watchers/Followers count',
        'square': 'Area (sqm)',
        'livingRoom': 'Living rooms count',
        'drawingRoom': 'Bedrooms count',
        'kitchen': 'Kitchens count',
        'bathRoom': 'Bathrooms count',
        'floor': 'Floor level',
        'buildingType': 'Building type',
        'constructionTime': 'Construction year',
        'renovationCondition': 'Renovation condition',
        'buildingStructure': 'Building structure',
        'ladderRatio': 'Units per elevator',
        'elevator': 'Has elevator (0/1)',
        'fiveYearsProperty': 'Ownership ≥ 5 years (0/1)',
        'subway': 'Near metro (0/1)',
        'district': 'District',
        'communityAverage': 'Community average price'
    }

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    input_values = {}

    for idx, feature in enumerate(original_features):
        col_idx = idx % 3
        with cols[col_idx]:
            desc = feature_descriptions.get(feature, feature)

            if model_loaded and feature in categorical_features:
                # Dropdown for categorical with reasonable cardinality
                unique_vals = df[feature].dropna().unique().tolist()
                if len(unique_vals) <= 20 and len(unique_vals) > 0:
                    input_values[feature] = st.selectbox(
                        f'{feature} ({desc})',
                        options=[''] + unique_vals,
                        key=f'input_{feature}'
                    )
                else:
                    input_values[feature] = st.text_input(
                        f'{feature} ({desc}) - Categorical',
                        value='',
                        key=f'input_{feature}'
                    )
            else:
                input_values[feature] = st.text_input(
                    f'{feature} ({desc})',
                    value='0',
                    key=f'input_{feature}'
                )

    if st.button('Make Prediction'):
        if model_loaded and pipeline is not None:
            try:
                input_df = pd.DataFrame([input_values])

                # Cast numerics
                for feat in numerical_features:
                    if feat in input_df.columns:
                        input_df[feat] = pd.to_numeric(input_df[feat], errors='coerce')

                # Predict
                pred = pipeline.predict(input_df)[0]
                st.success('Prediction Complete!')
                st.subheader('Predicted Total Price:')
                st.metric("Total Price", f"¥{pred:,.2f}")

                with st.expander("Show input summary"):
                    st.dataframe(input_df)

            except ValueError as e:
                st.error(f"Please enter valid values for all inputs. Error: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.info("Make sure all required fields are filled correctly.")
        else:
            st.error("Please train a model before making predictions.")

if __name__ == '__main__':
    main()
