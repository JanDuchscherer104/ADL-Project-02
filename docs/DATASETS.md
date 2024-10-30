# Datasets

1. [**Grocery Store Dataset**](https://paperswithcode.com/dataset/grocery-store):
Grocery Store is a dataset of natural images of grocery items. All natural images were taken with a smartphone camera in different grocery stores. It contains 5,125 natural images from 81 different classes of fruits, vegetables, and carton items (e.g. juice, milk, yoghurt). The 81 classes are divided into 42 coarse-grained classes, where e.g. the fine-grained classes 'Royal Gala' and 'Granny Smith' belong to the same coarse-grained class 'Apple'. Additionally, each fine-grained class has an associated iconic image and a product description of the item.
2. [**VEGFru**](https://paperswithcode.com/dataset/vegfru): VegFru is a domain-specific dataset for fine-grained visual categorization. VegFru categorizes vegetables and fruits according to their eating characteristics, and each image contains at least one edible part of vegetables or fruits with the same cooking usage. Particularly, all the images are labelled hierarchically. The current version covers vegetables and fruits of 25 upper-level categories and 292 subordinate classes. And it contains more than 160,000 images in total and at least 200 images for each subordinate class.
**Note**: _Not relevant_ as it contains images of veggies and fruits in a context, in which you would not find them in your own kitchen.

3. [**Food-101**](https://paperswithcode.com/dataset/food-101): The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101k images. The labels for the test images have been manually cleaned, while the training set contains some noise.
**Note**: _Not relevant for classification_ as it contains images of dishes and not individual ingredients.

4. [**Recipe1M**](https://paperswithcode.com/dataset/recipe1m-1): Contains images of dishes along with their recipes, helpful for both image classification and generating recipe-based articles.
![Recipe1M](/docs/assets/data/recipe1m.png)

**Note**: _Not relevant for classification_ as it contains images of dishes as well as a list of ingredients and instructions. However, this dataset might be usef for fine-tuning the LLM driving the agent.
