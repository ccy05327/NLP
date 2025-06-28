# Product Requirements Document (PRD) for AI Task Manager

## 1. Project Overview & Goal

- Project Title: Comparative Text Classification for Emotion Detection.

- Primary Goal: To produce a high-quality university coursework submission for the "CM3060 Natural Language Processing" module. The project must successfully implement, evaluate, and critically compare a traditional statistical text classification model against a modern embedding-based model to achieve a grade of 70% or higher.

- Target Persona (Developer): This project is for a solo university student with intermediate Python skills. The developer has access to a standard consumer-grade laptop (no enterprise-level GPUs).

- Core Mandate: The project must be a comparative study. The core technical exercise involves comparing the effectiveness of a traditional statistical model (like Logistic Regression) and a modern deep learning model (like DistilBERT) in addressing the chosen problem. The work must be situated within existing literature with proper citations.

## 2. Core Constraints & Technical Specifications

- Technology Stack: Python 3.9+

- Development Environment: Jupyter Notebook (.ipynb format).

- Key Libraries:

  - Data Handling: pandas, numpy

  - Machine Learning (Statistical): scikit-learn

  - Machine Learning (Embedding): huggingface-transformers, huggingface-datasets, pytorch or tensorflow

  - Visualisation: matplotlib, seaborn

- Dataset: dair-ai/emotion dataset, accessed via the huggingface-datasets library using the load_dataset function. It consists of English-language tweets labelled with one of six emotions: sadness, joy, love, anger, fear, or surprise.

- Deliverables:

    1. A final report in PDF format, written in British English, following the structure of the coursework brief.

    2. A single, clean, and fully commented Jupyter Notebook that contains all code for data loading, pre-processing, model training, evaluation, and visualisation. The notebook must be runnable from top to bottom without errors.

## 3. Feature Requirements & Work Breakdown Structure (WBS)

The project is broken down into four distinct Epics. Generate tasks based on these requirements.

### EPIC 1: Report Introduction & Scoping

- Requirement 1.1: Refine Introduction & Literature

  - User Story: As a researcher, I need to update the project introduction to reflect the comparative nature of the new project.

  - Acceptance Criteria:

    - The introduction must explicitly state the project's goal is to compare a classical statistical model vs. a modern embedding-based model.

    - The existing literature reference (Colnerič & Demšar, 2018) should be kept.

    - Find and integrate citations from 1-2 more recent academic papers (post-2020) that also perform this type of comparison for text classification tasks.

- Requirement 1.2: Update Project Objectives

  - User Story: As a student, I need to rewrite the project objectives to align with the comparative coursework brief.

  - Acceptance Criteria:

    - The new primary objective must focus on the comparison of Logistic Regression and a transformer model.

    - Specific objectives must include implementing both pipelines and analysing their trade-offs (performance, computational cost, interpretability).

- Requirement 1.3: Finalise Evaluation Methodology

  - User Story: As a student, I need to confirm that my evaluation metrics are suitable for comparing both models.

  - Acceptance Criteria:

    - The section on evaluation metrics (Accuracy, Precision, Recall, F1-Score, Macro/Weighted Averages) is confirmed as appropriate.

    - The text must be updated to explicitly state these metrics will be used for direct quantitative comparison between the two main models.

### EPIC 2: Code Implementation & Development

- Requirement 2.1: Implement Data Processing Pipelines

  - User Story: As a developer, I need to create two distinct data pre-processing pipelines, one for each model type, within my Jupyter Notebook.

  - Acceptance Criteria:

    - Pipeline A (Statistical): Create a pipeline that takes the raw text from the emotion dataset and uses scikit-learn's TfidfVectorizer to convert it into numerical features. This pipeline will feed into the Logistic Regression model.

    - Pipeline B (Embedding): Create a pipeline that uses a DistilBertTokenizer from the transformers library to tokenize the text, add special tokens, and create input_ids and attention_mask tensors. This pipeline will feed into the DistilBERT model.

- Requirement 2.2: Implement & Evaluate Statistical Model

  - User Story: As a developer, I need to implement and train a Logistic Regression model and a Naïve Bayes baseline using the TF-IDF processed data.

  - Acceptance Criteria:

    - The Naïve Bayes model is clearly defined and evaluated as the official "baseline" model.

    - The Logistic Regression model is defined as the primary "statistical model".

    - Both models are trained and evaluated on the test set. Performance metrics (Accuracy, F1-score) are calculated and saved.

- Requirement 2.3: Implement & Evaluate Embedding-based Model

  - User Story: As a developer, I need to fine-tune a pre-trained DistilBERT model for emotion classification.

  - Acceptance Criteria:

    - Load a pre-trained distilbert-base-uncased model using the AutoModelForSequenceClassification class from transformers.

    - Fine-tune the model on the training portion of the emotion dataset using the Hugging Face Trainer API for simplicity.

    - Evaluate the fine-tuned model on the test set. Performance metrics are calculated. A confusion matrix is generated using seaborn.

### EPIC 3: Analysis & Report Writing

- Requirement 3.1: Write Comparative Performance Analysis

  - User Story: As an analyst, I need to write the core evaluation section of the report, directly comparing the performance of the Logistic Regression and DistilBERT models.

  - Acceptance Criteria:

    - A summary table comparing the Accuracy, Macro F1-Score, and Weighted F1-Score of all three models (Naïve Bayes, Logistic Regression, DistilBERT) is created.

    - The confusion matrices for Logistic Regression and DistilBERT are presented side-by-side for easy comparison.

    - A detailed written discussion is produced, analysing why the performance differences exist (e.g., semantic understanding of embeddings vs. word frequency of TF-IDF), referencing specific misclassifications from the confusion matrices.

- Requirement 3.2: Write Project Summary & Reflections

  - User Story: As a student, I need to write the final reflective conclusion for the report.

  - Acceptance Criteria:

    - The conclusion must summarise the project's key finding (i.e., that transformers outperform statistical models at a higher computational cost).

    - It must reflect on the practical trade-offs (e.g., when one might prefer the simpler model due to efficiency or interpretability).

    - It must discuss the reproducibility of the work and suggest future improvements (e.g., using larger models, different data augmentation

### EPIC 4: Finalisation & Submission

- Requirement 4.1: Final Code and Report Assembly

  - User Story: As a student, I need to assemble my final deliverables for submission.

  - Acceptance Criteria:

    - The Jupyter Notebook is cleaned of all experimental/messy code, fully commented to explain each step, and tested to run from start to finish without errors.

    - All written sections are collated into a single document, proofread for grammatical errors and clarity, and formatted according to academic standards.

    - The final report is exported as a PDF.

## 4. Definition of Done

The project is considered "Done" when:

1. All requirements listed above have been met.

2. The final PDF report and the .ipynb notebook have been created.

3. The report and code fully address all sections of the coursework rubric, especially the comparative analysis requirements.

4. The work is ready for submission.
