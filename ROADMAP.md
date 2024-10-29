### ROADMAP.md

1. **Project Overview**
   - The main objective of this project is to build a recipe-generating app powered by image classifiers, LLMs, and diffusion models. The app takes input images from a webcam, classifies the ingredients, generates textual content, and enhances the output with diffusion model-generated visuals. This is an end-to-end system involving data capture, classification, content generation, and visual augmentation.
   - The system must be robust, modular, and capable of handling real-time data input from the webcam to final article output, which includes both text and visuals. Key components include: Dataset selection, classifier models (AlexNet, ResNet, ViT), LLM toolchain (LangChain), diffusion models, integrations for experiment tracking (W&B), hyperparameter optimization (Optuna), and interpretability tools (Captum).

2. **Design Patterns**
   - **Factory Pattern**: Used for instantiating models dynamically based on requirements (e.g., AlexNet, ResNet, ViT) and configurable settings through Pydantic. This ensures flexibility in switching between different architectures as needed.
   - **Singleton Pattern**: Ensures a single instance for shared resources, such as `PathConfig` and `_SharedParams`. This pattern manages global settings, minimizing redundant reconfigurations and ensuring that different modules operate on consistent paths and settings.
   - **Facade Pattern**: Simplifies interaction with complex subsystems like W&B, Optuna, and MLflow by encapsulating the complexity behind a simple interface (`TrainerFactory`). This improves code readability and manageability.
   - **Observer Pattern**: Used for logging and tracking hyperparameters and metrics in real-time using W&B and MLflow, thus ensuring all relevant components are updated with the latest results for effective experiment tracking and reproducibility.

3. **Dataset Selection and Setup**
   - Potential datasets include Food-101, Grocery Store, VEGFru, and Recipe1M. Dataset selection will focus on relevance to food classification, diversity, ease of use, and potential to enhance accuracy when extended with self-curated data (`ds2` via autodistill).
   - Develop a PyTorch `Dataset` class and implement a `LightningDataModule` for efficient data loading and handling. The data must be augmented using Albumentations to improve generalizability—augmentations include random crops, rotations, brightness changes, flips, and noise addition.
   - **Tools**: PyTorch, PyTorch Lightning, Albumentations, autodistill (for generating labeled data). Documentation will clearly define input size, augmentations used, data splits, and preprocessing steps.

4. **Core Components**
   - **Webcam Module**: Implements data acquisition using `opencv-python`. The module captures images based on user prompts and feeds them into the classifier pipeline. **Tools**: OpenCV.
     - **Assigned To**: Jan
     - **Deadline**: November 15, 2024
   - **Image Classifier**: The classification pipeline includes model selection, training, validation, and evaluation of AlexNet (built from scratch), ResNet-50, and Vision Transformer (fine-tuned pre-trained models). Training results are logged on W&B.
     - **Tasks**: Implement models, handle dataset integration, implement fine-tuning, and optimize performance.
     - **Tools**: PyTorch, PyTorch Lightning, W&B, Captum for model interpretability, Optuna for hyperparameter tuning.
     - **Assigned To**: Jan
     - **Deadline**: December 5, 2024
     - **Note**: Confirm correct instantiation of ViT and ResNet with ImageNet weights, replace their classifier heads, and prepare them for finetuning.
   - **LLM Toolchain**: Utilizes LangChain to process classified labels into coherent recipe content. Integrates external information from Wikipedia and DuckDuckGo to enhance textual descriptions, producing a rich content base for articles.
     - **Tools**: LangChain, Wikipedia API, DuckDuckGo.
     - **Assigned To**: Flo
     - **Deadline**: December 20, 2024
   - **Diffusion Models**: Generate high-quality images based on textual content provided by the LLM. This model augments the recipe output by generating visuals that make the articles more engaging.
     - **Tools**: Stable Diffusion, Python API for model calls.
     - **Assigned To**: Flo
     - **Deadline**: January 5, 2025
   - **Article Assembler**: Combines the LLM-generated text and diffusion model images into a structured article format. Uses Markdown as the intermediary format and converts it into a PDF using Pandoc for easy consumption.
     - **Tools**: Markdown, Pandoc.
     - **Assigned To**: Jan
     - **Deadline**: January 20, 2025

5. **Milestones**
   - **Data Preprocessing and Augmentation**: Complete augmentation strategies, create `Dataset` and `LightningDataModule`. **Deadline**: Early November 2024. **Assigned To**: Jan.
   - **Webcam Module Implementation**: Build and integrate the webcam module to collect live images. **Deadline**: November 15, 2024. **Assigned To**: Jan.
   - **Initial Model Training**: Train AlexNet and log initial benchmarks. Start fine-tuning pre-trained models (ResNet-50 and ViT) with ImageNet weights. Track results using W&B. **Deadline**: December 5, 2024. **Assigned To**: Jan.
   - **LLM Integration**: Develop an interface using LangChain to gather and generate recipe content. **Deadline**: December 20, 2024. **Assigned To**: Flo.
   - **Diffusion Model Integration**: Implement the diffusion model to enhance article visuals. **Deadline**: January 5, 2025. **Assigned To**: Flo.
   - **System Integration and Testing**: Test the overall flow from webcam capture to article PDF generation, ensuring all components work seamlessly. **Deadline**: End of January 2025. **Assigned To**: Jan and Flo.