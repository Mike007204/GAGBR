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

def bootstrap_shap_confidence_intervals(shap_values, feature_names, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for mean absolute SHAP values
    """
    n_samples = shap_values.shape[0]
    n_features = shap_values.shape[1]
    
    # Store bootstrap results
    bootstrap_means = np.zeros((n_bootstrap, n_features))
    
    # Perform bootstrap sampling
    np.random.seed(42)  # For reproducibility
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sampled_shap = shap_values[indices]
        
        # Calculate mean absolute SHAP for this bootstrap sample
        bootstrap_means[i] = np.mean(np.abs(sampled_shap), axis=0)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    results = {}
    for i, feature in enumerate(feature_names):
        lower_ci = np.percentile(bootstrap_means[:, i], lower_percentile)
        upper_ci = np.percentile(bootstrap_means[:, i], upper_percentile)
        mean_val = np.mean(bootstrap_means[:, i])
        std_val = np.std(bootstrap_means[:, i])
        
        results[feature] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci
        }
    
    return results, bootstrap_means

def main():
    st.title('GA-GBR for Property Valuation🏆🚀')

    st.subheader('Demonstration:')
    st.subheader('1) Raw dataset')
    st.subheader('2) Model hyperparameters and Evaluation Metrics')
    st.subheader('3) SHAP: Feature importance with Bootstrap Confidence Intervals')
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
    
    # Add bootstrap options
    st.sidebar.subheader('Bootstrap Analysis Options')
    perform_bootstrap = st.sidebar.checkbox('Perform Bootstrap Analysis for SHAP', value=True)
    n_bootstrap = st.sidebar.slider('Number of Bootstrap Iterations', 100, 2000, value=1000, step=100)
    confidence_level = st.sidebar.slider('Confidence Level', 0.90, 0.99, value=0.95, step=0.01)

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
        
        # ---- Bootstrap Analysis for SHAP Confidence Intervals ----
        if perform_bootstrap:
            st.write('---')
            st.subheader('📊 Bootstrap Confidence Intervals for SHAP Values')
            
            with st.spinner(f'Performing bootstrap analysis ({n_bootstrap} iterations)...'):
                # Use test set for bootstrap analysis
                X_test_sample = X_test[:min(200, X_test.shape[0])]
                shap_values_test = explainer.shap_values(X_test_sample)
                
                # Get the actual feature names for the processed data
                actual_feature_names = feature_names_processed[:X_test_sample.shape[1]]
                
                bootstrap_results, bootstrap_means = bootstrap_shap_confidence_intervals(
                    shap_values_test, 
                    actual_feature_names, 
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level
                )
            
            # Display results
            st.success(f'Bootstrap analysis complete! ({n_bootstrap} iterations)')
            
            # Sort features by mean importance
            sorted_features = sorted(bootstrap_results.items(), 
                                     key=lambda x: x[1]['mean'], 
                                     reverse=True)
            
            # Create DataFrame for display
            results_data = []
            for i, (feature, stats) in enumerate(sorted_features[:15], 1):
                results_data.append({
                    'Rank': i,
                    'Feature': feature,
                    'Mean |SHAP|': stats['mean'],
                    'Std Dev': stats['std'],
                    f'{int(confidence_level*100)}% CI Lower': stats['ci_lower'],
                    f'{int(confidence_level*100)}% CI Upper': stats['ci_upper'],
                    'CI Width': stats['ci_upper'] - stats['ci_lower']
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Display top features table
            st.write(f"**Top 15 Features by Mean |SHAP| with {int(confidence_level*100)}% Confidence Intervals:**")
            st.dataframe(results_df.style.format({
                'Mean |SHAP|': '{:.4f}',
                'Std Dev': '{:.4f}',
                f'{int(confidence_level*100)}% CI Lower': '{:.4f}',
                f'{int(confidence_level*100)}% CI Upper': '{:.4f}',
                'CI Width': '{:.4f}'
            }))
            
            
            # Visualize bootstrap distributions for top 6 features
            st.write("**Bootstrap Distributions for Top 6 Features:**")
            fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (feature, _) in enumerate(sorted_features[:6]):
                ax = axes[idx]
                feature_idx = actual_feature_names.index(feature)
                
                # Plot histogram
                ax.hist(bootstrap_means[:, feature_idx], bins=30, alpha=0.7, 
                        color='skyblue', edgecolor='black')
                
                # Add CI and mean lines
                ci_lower = bootstrap_results[feature]['ci_lower']
                ci_upper = bootstrap_results[feature]['ci_upper']
                mean_val = bootstrap_results[feature]['mean']
                
                ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                          label=f'{int(confidence_level*100)}% CI')
                ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.axvline(mean_val, color='darkgreen', linewidth=2, label='Mean')
                
                # Add text annotation
                ax.text(0.98, 0.98, f'Mean: {mean_val:.3f}\nCI: [{ci_lower:.3f}, {ci_upper:.3f}]',
                        transform=ax.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Truncate long feature names for display
                display_name = feature if len(feature) <= 20 else feature[:17] + '...'
                ax.set_title(f'{display_name}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Mean |SHAP|')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                if idx == 0:
                    ax.legend(loc='upper left', fontsize=8)
            
            plt.suptitle(f'Bootstrap Distributions of Mean |SHAP| Values ({n_bootstrap} iterations)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.clf()
            
            # Export option
            csv_export = results_df.to_csv(index=False)
            st.download_button(
                label="Download Bootstrap Results as CSV",
                data=csv_export,
                file_name='shap_bootstrap_results.csv',
                mime='text/csv'
            )
            
            # Statistical summary
            st.write("**Statistical Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Feature Mean |SHAP|", 
                         f"{sorted_features[0][1]['mean']:.4f}")
            with col2:
                avg_ci_width = np.mean([stats['ci_upper'] - stats['ci_lower'] 
                                        for _, stats in sorted_features[:10]])
                st.metric("Avg CI Width (Top 10)", f"{avg_ci_width:.4f}")


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
                
                # Display price with 10k (万元) unit
                price_in_10k = pred * 10000
                st.metric("Total Price", f"{price_in_10k:,.0f} (10k) ¥")

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
