# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classification model that predicts whether an individual’s annual income is greater than \$50,000 or less than or equal to \$50,000 based on US Census data. The model uses scikit-learn’s `RandomForestClassifier` and is trained via the `train_model.py` script.

Input features include age, workclass, education, marital status, occupation, relationship, race, sex, native country, capital gains, capital losses, and hours worked per week.

This model was developed as part of a Udacity/WGU project to deploy a scalable machine learning pipeline with FastAPI.

## Intended Use

This is designed to:
- Build an end-to-end machine learning pipeline, including pre-processing, model training, evaluation, and deployment.
- Integrate model training and inference into an API using FastAPI.
- Monitor a model's performance on slices of data (e.g., by race, sex, or education level).

## Training Data

The model is trained on the `census.csv` dataset included in this project, which is derived from the Adult/Census Income dataset. Each row corresponds to an individual, and the label (`salary`) indicates whether that individual’s income is `>50K` or `<=50K`.

For training, the full dataset is randomly split into training and test sets using an 80/20 split. The training split is passed through the `process_data` function, which:
- Separates the label column (`salary`) from the other features.
- Applies one-hot encoding to the categorical columns (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`).
- Leaves the continuous features as numeric values (`age`, `education-num`, `capital-gain`, `capital-loss`, and `hours-per-week`).
- Binarizes the label into 0/1 values.

The resulting arrays (`X_train` and `y_train`) are used to fit the random forest model.

## Evaluation Data

The evaluation data consists of the 20% test split from the `census.csv` data.

This test set is processed with the same `process_data` function. This ensures consistent encoding between training and evaluation. The processed test features (`X_test`) and labels (`y_test`) are used to understand the model's overall performance.

The model’s performance is evaluated on **data slices**. Metrics for each slice are computed using the `performance_on_categorical_slice` function and written to `slice_output.txt`.

## Metrics

The primary evaluation metrics for this model are **precision**, **recall**, and **F1 score** (beta = 1):
- **Precision** is the proportion of predicted `>50K` incomes that are actually `>50K`.
- **Recall** is the proportion of actual `>50K` incomes that are correctly identified by the model.
- **F1 score** is the mean of both precision and recall (a balance of both types of errors).

The model achieved the following overall performance on the test set:
- **Precision**: 0.7419  
- **Recall**: 0.6384  
- **F1 score**: 0.6863  

In addition to the aggregate metrics, slice-level metrics are computed and logged in `slice_output.txt`:
- By `education`, the model tends to perform better on more highly educated groups (F1 is **0.7404** for `Bachelors` (1,053 samples), **0.8409** for `Masters` (369 samples), **0.8793** for `Doctorate` (77 samples), and **0.8852** for `Prof-school` (116 ). In contrast, performance is weaker for some lower-education categories, such as `HS-grad` (F1 **0.5261**), `Some-college` (F1 **0.5914**), and especially `10th` (F1 **0.2353**) and `7th-8th` (F1 **0.0000`).
- Results by `race` and `sex` also show variation. For `race`, the F1 score is **0.6850** for `White` (5,595 samples), **0.6667** for `Black` (599 samples), and **0.7458** for `Asian-Pac-Islander` (193 samples). For `sex`, the F1 score is **0.6997** for `Male` (4,387 samples) versus **0.6015** for `Female` (2,126 samples).
- By `native-country`, the largest group, `United-States` (5,870 samples), has an F1 of **0.6814**, while many small-country slices show extreme scores (**0.0000** or **1.0000**). These, however, are based on very few samples.

So, based on aggregate metrics, the model performs reasonably well, however the slice-level metrics indicate the model’s behavior is not uniform across demographic groups. Small-sample slices can produce misleadingly high or low scores.

## Ethical Considerations

This model is trained on census data that includes socially sensitive demographic attributes such as race, sex, marital status, and native country. The model and its predictions may reflect historical and societal biases that may be present in the data. This could lead to inadvertent amplification of existing inequities and biases, especially in contexts such as hiring, lending, or access to opportunities, etc..

## Caveats and Recommendations

This model has several important limitations:
- It is part of a learning project and has not been optimized or validated for deployment in production.
- The model is trained and evaluated on a single dataset and a single train/test split. Performance **will** vary with different data.
- No hyperparameter tuning, calibration, or optimization have been applied. The RandomForestClassifier may not be the best or fairest model for this usage.
- Slice-level metrics reveal that performance is not uniform across groups.

Recommendations:
- This model is for education only and should not be used in production.
