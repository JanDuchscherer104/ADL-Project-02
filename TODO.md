### TODO.md

1. **Immediate Tasks**
   - **Dataset Selection (Assigned to Jan)**:
     - Decide on a primary dataset from the listed options (Food-101, VEGFru, Recipe1M). Criteria include diversity, image quality, relevance to food classification, and potential for extension with self-curated data. Use Google Dataset Search or Kaggle for additional options.
     - **Tools**: Google Dataset Search, Kaggle.
     - **Deadline**: November 1, 2024
   - **PyTorch Dataset and DataModule (Assigned to Jan)**:
     - Create a custom `Dataset` class for the selected dataset. Integrate data transformations using Albumentations for augmentations like random crops, rotations, flips, and color jitter.
     - Implement a `LightningDataModule` to manage data splits for training, validation, and testing.
     - **Tools**: PyTorch, Albumentations, PyTorch Lightning.
     - **Deadline**: November 10, 2024

2. **Webcam Module and Initial Model Training**
   - **Webcam Module (Assigned to Jan)**:
     - Implement a `WebcamCapture` module using `opencv-python` for capturing images. Ensure that captured images are appropriately processed for classification.
     - Integrate a command-line trigger to make the webcam capture user-friendly.
     - **Tools**: OpenCV, Python CLI.
     - **Deadline**: November 15, 2024
   - **Image Classifier Training (Assigned to Jan)**:
     - **AlexNet Implementation**: Implement AlexNet from scratch and conduct initial training runs to establish baseline metrics. Verify with PyTorch Lightning for streamlined training and validation.
     - **Pre-trained Models**: Fine-tune ResNet-50 and Vision Transformer on both `ds1` and `ds2`. Include hyperparameter optimization using Optuna for efficient tuning.
       - **Tasks**:
         - Integrate ImageNet pre-trained weights for ResNet-50 and Vision Transformer.
         - Replace their classifier heads to match the number of classes in the dataset.
         - Prepare the models for fine-tuning.
         - Integrate and configure W&B for experiment logging.
         - Optimize configurations like learning rate, batch size, and epochs.
         - Use Optuna to automate hyperparameter search.
       - **Tools**: PyTorch, PyTorch Lightning, Optuna, W&B.
       - **Deadline**: December 5, 2024

3. **Testing, Evaluation, and Interpretability**
   - **Model Testing (Assigned to Jan)**:
     - Implement unit tests for data transformations and model training routines. Confirm consistency across training, validation, and test phases.
     - **Tools**: PyTest.
     - **Deadline**: December 10, 2024
   - **Model Evaluation (Assigned to Jan)**:
     - **Metrics Calculation**: Track metrics including accuracy, precision, recall, and F1 score for each model. Compare across `ds1` and `ds2` datasets.
     - **Training Analysis**: Use PyTorch Lightning's profiler to assess training bottlenecks. Log training loop performance and metrics to W&B for transparency.
     - **Captum Analysis**: Analyze best and worst case predictions to assess model decision-making. Identify failure cases and iterate on model improvements.
     - **Tasks**:
       - Visualize Loss Curves.
       - Analyze Training Loop Profiler Results.
       - Conduct Best & Worst Case Analysis using Captum.
     - **Tools**: PyTorch Lightning Profiler, Captum, W&B.
     - **Deadline**: December 20, 2024

4. **Article Generation Components**
   - **LLM Integration (Assigned to Flo)**:
     - Utilize LangChain to convert classified image labels into descriptive recipe texts. Enrich content with information gathered from Wikipedia and DuckDuckGo.
     - **Tasks**: Develop the LangChain integration and test API interactions with Wikipedia and DuckDuckGo.
     - **Input**: `List[ImageLabels]`.
     - **Output**: Recipe Text.
     - **Tools**: LangChain, Wikipedia API, DuckDuckGo API.
     - **Deadline**: December 20, 2024
   - **Diffusion Model Integration (Assigned to Flo)**:
     - Connect diffusion models to generate images for recipes. Use outputs from the LLM toolchain to describe the kind of visuals needed.
     - **Tasks**: Test image generation quality and relevance.
     - **Tools**: Stable Diffusion API, Python.
     - **Deadline**: January 5, 2025

5. **Article Assembly (Assigned to Jan)**
   - **Markdown Template Creation**:
     - Create structured Markdown templates to house the recipe text and visuals. Ensure each article meets content requirements (minimum 4 paragraphs, 4 figures).
     - **Tools**: Markdown, Jinja2 (for templating).
     - **Deadline**: January 10, 2025
   - **PDF Generation**:
     - Convert assembled Markdown content to a formatted PDF using Pandoc. Ensure formatting and content integrity during the conversion process.
     - **Tasks**: Automate PDF generation from Markdown and conduct visual quality checks.
     - **Tools**: Pandoc, Python.
     - **Deadline**: January 20, 2025

6. **Documentation and Future Roadmap**
