# **What is AI?**
Simply put, AI is software that imitates human behaviors and capabilities. Key workloads include:

1. Machine learning - This is often the foundation for an AI system, and is the way we "teach" a computer model to make predictions and draw conclusions from data.
2. Computer vision - Capabilities within AI to interpret the world visually through cameras, video, and images.
3. Natural language processing - Capabilities within AI for a computer to interpret written or spoken language, and respond in kind.
4. Document intelligence - Capabilities within AI that deal with managing, processing, and using high volumes of data found in forms and documents.
5. Knowledge mining - Capabilities within AI to extract information from large volumes of often unstructured data to create a searchable knowledge store.
6. Generative AI - Capabilities within AI that create original content in a variety of formats including natural language, image, code, and more.

**Azure Machine Learning Studio offers multiple authoring experiences such as:**
1. Automated machine learning: this feature enables non-experts to quickly create an effective machine learning model from data.
2. Azure Machine Learning designer: a graphical interface enabling no-code development of machine learning solutions.
3. Data metric visualization: analyze and optimize your experiments with visualization.
4. Notebooks: write and run your own code in managed Jupyter Notebook servers that are directly integrated in the studio.

# **Understand computer vision**
1. Image classification - Image classification involves training a machine learning model to **classify images based on their contents**.
2. Object detection - Object detection machine learning models are trained to classify individual objects within an image, and **identify their location with a bounding box**.
3. Semantic segmentation - Semantic segmentation is an advanced machine learning technique in which **individual pixels in the image are classified according to the object to which they belong**.
4. Image analysis - You can create solutions that combine machine learning models with advanced image analysis techniques to extract information from images, including "tags" that could help catalog the image or even **descriptive captions that summarize the scene shown in the image**.
5. Face detection, analysis, and recognition - Face detection is a specialized form of object detection that locates human faces in an image. This can be combined with classification and facial geometry analysis techniques to recognize individuals based on their **facial features**.
6. Optical character recognition (OCR) - Optical character recognition is a technique used to **detect and read text in image**

You can use Microsoft's Azure AI Vision to develop computer vision solutions.
Some features of Azure AI Vision include:
1. Image Analysis: capabilities for analyzing images and video, and extracting descriptions, tags, objects, and text.
2. Face: capabilities that enable you to build face detection and facial recognition solutions.
3. Optical Character Recognition (OCR): capabilities for extracting printed or handwritten text from images, enabling access to a digital version of the scanned text.
-----
One of the most common machine learning model architectures for computer vision is a **convolutional neural network (CNN)**, a type of deep learning architecture. CNNs use filters to extract numeric feature maps from images, and then feed the feature values into a deep learning model to generate a label prediction.

Transformers and multi-modal models

**Transformers** work by processing huge volumes of data, and encoding language tokens (representing individual words or phrases) as vector-based embeddings (arrays of numeric values). You can think of an embedding as representing a set of dimensions that each represent some semantic attribute of the token. The embeddings are created such that tokens that are commonly used in the same context are closer together dimensionally than unrelated words.

The success of transformers as a way to build language models has led AI researchers to consider whether the same approach would be effective for image data. The result is the development of **multi-modal models**, in which the model is trained using a large volume of captioned images, with no fixed labels. An image encoder extracts features from images based on pixel values and combines them with text embeddings created by a language encoder. The overall model encapsulates relationships between natural language token embeddings and image features, as shown here:

![image](https://github.com/user-attachments/assets/b81e6883-e99d-4a58-93aa-133588a67554)

The Microsoft Florence model is just such a model. Trained with huge volumes of captioned images from the Internet, it includes both a language encoder and an image encoder. 

![image](https://github.com/user-attachments/assets/55cb0433-ddcf-4df9-8bf4-88a9d3c81e5f)

Azure resources for Azure AI Vision service
1. **Azure AI Vision**
2. **Azure AI services**: includes Azure AI Vision along with many other Azure AI services; such as Azure AI Language, Azure AI Custom Vision, Azure AI Translator, and others

Azure AI Vision supports multiple image analysis capabilities, including:
1. Optical character recognition (OCR) - extracting text from images.
2. Generating captions and descriptions of images.
3. Detection of thousands of common objects in images.
4. Tagging visual features in images
These tasks, and more, can be performed in **Azure AI Vision Studio**
-----
**Face detection** involves identifying regions of an image that contain a human face, typically by returning bounding box coordinates that form a rectangle around the face

Microsoft Azure provides multiple Azure AI services that you can use to detect and analyze faces, including:
1. **Azure AI Vision**, which offers face detection and some basic face analysis, such as returning the bounding box coordinates around an image.
2. **Azure AI Video Indexer**, which you can use to detect and identify faces in a video.
3. **Azure AI Face**, which offers pre-built algorithms that can detect, recognize, and analyze faces.
Of these, Face offers the widest range of facial analysis capabilities.

There are some considerations that can help improve the accuracy of the detection in the images:
1. Image format - supported images are JPEG, PNG, GIF, and BMP.
2. File size - 6 MB or smaller.
3. Face size range - from 36 x 36 pixels up to 4096 x 4096 pixels. Smaller or larger faces will not be detected.
4. Other issues - face detection can be impaired by extreme face angles, extreme lighting, and occlusion (objects blocking the face such as a hand).
you can use **Vision Studio** to explore the capabilities of Azure AI Vision.

**Responsible AI use**
To support Microsoft's Responsible AI Standard, Azure AI Face and Azure AI Vision have a Limited Access policy.
Anyone can use the Face service to:
1. Detect the location of faces in an image.
2. Determine if a person is wearing glasses.
3. Determine if there's occlusion, blur, noise, or over/under exposure for any of the faces.
4. Return the head pose coordinates for each face in an image.

The Limited Access policy requires customers to submit an intake form to access additional Azure AI Face service capabilities including:
1. **Face verification**: the ability to compare faces for similarity.
2. **Face identification**: the ability to identify named individuals in an image.
3. **Liveness detection**: the ability to detect and mitigate instances of recurring content and/or behaviors that indicate a violation of policies (eg. such as if the input video stream is real or fake).

Azure resources for Face
1. **Face**: track utilization and costs for Face separately.
2. **Azure AI services**: with many other Azure AI services such as Azure AI Content Safety, Azure AI Language, and others.
-----
Optical character recognition (OCR) enables artificial intelligence (AI) systems to read text in images, enabling applications to extract information from photographs, scanned documents, and other sources of digitized text.

**Uses of OCR**
Automating text processing can improve the speed and efficiency of work by removing the need for manual data entry. The ability to recognize printed and handwritten text in images is beneficial in scenarios such as note taking, digitizing medical records or historical documents, scanning checks for bank deposits, and more.

The ability for computer systems to process written and printed text is an area of AI where computer vision intersects with **natural language processing**. Vision capabilities are needed to "read" the text, and then natural language processing capabilities make sense of it. Azure AI Vision's Read API is the OCR engine that powers text extraction from **images, PDFs, and TIFF files**. **The Read API, otherwise known as Read OCR engine**, uses the latest recognition models and is optimized for images that have a significant amount of text or have considerable visual noise. 

Calling the Read API returns results arranged into the following hierarchy:
Pages - One for each page of text, including information about the page size and orientation.
Lines - The lines of text on a page.
Words - The words in a line of text, including the bounding box coordinates and text itself.

![image](https://github.com/user-attachments/assets/85b81319-4352-4f5a-8961-668eeb6181bd)

Azure resources for AI Vision
1. Azure AI Vision
2. Azure AI Services
Once you've created a resource, there are several ways to use Azure AI Vision's Read API:
1. Vision Studio
2. REST API
3. Software Development Kits (SDKs): Python, C#, JavaScript


# **Understand natural language processing**
NLP enables you to create software that can:

1. Analyze and interpret text in documents, email messages, and other sources.
2. Interpret spoken language, and synthesize speech responses.
3. Automatically translate spoken or written phrases between languages.
4. Interpret commands and determine appropriate actions.

You can use Microsoft's **Azure AI Language** to build natural language processing solutions. Some features of Azure AI Language include understanding and analyzing text, training conversational language models that can understand spoken or text-based commands, and building intelligent applications.
Microsoft's **Azure AI Speech** is another service that can be used to build natural language processing solutions. Azure AI Speech features include speech recognition and synthesis, real-time translations, conversation transcriptions, and more.
Microsoft's **Azure AI Translator** uses a **Neural Machine Translation (NMT) model for translation**, which analyzes the semantic context of the text and renders a more accurate and complete translation as a result.
You can explore Azure AI Language features in the Azure Language Studio and Azure AI Speech features in the Azure Speech Studio. The service features are available for use and testing in the studios and other programming languages.

# **Document intelligence in Microsoft Azure**
You can use Microsoft's **Azure AI Document Intelligence** to build solutions that manage and accelerate data collection from scanned documents. Features of Azure AI Document Intelligence help automate document processing in applications and workflows, enhance data-driven strategies, and enrich document search capabilities. The service features are available for use and testing in the **Document Intelligence Studio and other programming languages**.
**Knowledge mining** is the term used to describe solutions that involve extracting information from large volumes of often unstructured data to create a searchable knowledge store. One Microsoft knowledge mining solution is **Azure AI Search**, a private, enterprise, search solution that has tools for building indexes.

# **Generative AI in Microsoft Azure**
**Azure OpenAI Service** is Microsoft's cloud solution for deploying, customizing, and hosting generative AI models. The service features are available for use and testing with Azure AI Foundry, Microsoft's platform for designing enterprise-grade AI solutions. You can use the **Azure AI Foundry** portal to manage, develop, and customize generative AI models.

**Understand Responsible AI**
1. Fairness
2. Reliability and safety
3. Privacy and security
4. Inclusiveness
5. Transparency
6. Accountability
