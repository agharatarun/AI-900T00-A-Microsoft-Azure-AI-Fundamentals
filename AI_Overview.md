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
https://portal.vision.cognitive.azure.com/gallery/featured

1. Image classification - Image classification involves training a machine learning model to **classify images based on their contents**.
2. Object detection - Object detection machine learning models are trained to classify individual objects within an image, and **identify their location with a bounding box**.
3. Semantic segmentation - Semantic segmentation is an advanced machine learning technique in which **individual pixels in the image are classified according to the object to which they belong**.
4. Image analysis - You can create solutions that combine machine learning models with advanced image analysis techniques to extract information from images, including "tags" that could help catalog the image or even **descriptive captions that summarize the scene shown in the image**.
5. Face detection, analysis, and recognition - Face detection is a specialized form of object detection that locates human faces in an image. This can be combined with classification and facial geometry analysis techniques to recognize individuals based on their **facial features**.
6. Optical character recognition (OCR) - Optical character recognition is a technique used to **detect and read text in image**

Classification is used to predict categories of data. It can predict which category or class an item of data belongs to. In this example, a machine learning model trained by using classification with labeled data can be used to determine the type of bone fracture in a new scan that is not labeled already. Featurization is not a machine learning type. Regression is used to predict numeric values. Clustering analyzes unlabeled data to find similarities in the data.

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
https://language.cognitive.azure.com

NLP enables you to create software that can:
1. Analyze and interpret text in documents, email messages, and other sources.
2. Interpret spoken language, and synthesize speech responses.
3. Automatically translate spoken or written phrases between languages.
4. Interpret commands and determine appropriate actions.

You can use Microsoft's **Azure AI Language** to build natural language processing solutions. Some features of Azure AI Language include understanding and analyzing text, training conversational language models that can understand spoken or text-based commands, and building intelligent applications.
Microsoft's **Azure AI Speech** is another service that can be used to build natural language processing solutions. Azure AI Speech features include speech recognition and synthesis, real-time translations, conversation transcriptions, and more.
Microsoft's **Azure AI Translator** uses a **Neural Machine Translation (NMT) model for translation**, which analyzes the semantic context of the text and renders a more accurate and complete translation as a result.
You can explore Azure AI Language features in the Azure Language Studio and Azure AI Speech features in the Azure Speech Studio. The service features are available for use and testing in the studios and other programming languages.

-----
**Text Analysis with the Language Service**

Azure AI Language is a cloud-based service that includes features for understanding and analyzing text. Azure AI Language includes various features that support sentiment analysis, key phrase identification, text summarization, and conversational language understanding.

Frequency analysis: **Term frequency - inverse document frequency (TF-IDF)** is a common technique in which a score is calculated based on how often a word or term appears in one document compared to its more general frequency across the entire collection of documents. Using this technique, a high degree of relevance is assumed for words that appear frequently in a particular document, but relatively infrequently across a wide range of other documents.

text classification: Another useful text analysis technique is to use a classification algorithm, such as logistic regression, to train a machine learning model that classifies text based on a known set of categorizations. A common application of this technique is to train a model that classifies text as positive or negative in order to perform sentiment analysis or opinion mining.

![image](https://github.com/user-attachments/assets/20e66364-c096-4219-942b-afc6d7cdb3b9)

Azure AI Language is a part of the Azure AI services offerings that can perform advanced natural language processing over unstructured text. Azure AI Language's **text analysis** features include:

1. Named entity recognition identifies people, places, events, and more. This feature can also be customized to extract custom categories. Named Entity Recognition (NER) is the ability to identify different entities in text and categorize them into pre-defined classes or types such as: person, location, event, product, and organization.
In this question, the square brackets indicate the entities such as DateTime, PersonType, Skill.
2. Entity linking identifies known entities together with a link to Wikipedia.
3. Personal identifying information (PII) detection identifies personally sensitive information, including personal health information (PHI).
4. Language detection identifies the language of the text and returns a language code such as "en" for English.
5. Sentiment analysis and opinion mining identifies whether text is positive or negative.
6. Summarization summarizes text by identifying the most important information.
6. Key phrase extraction lists the main concepts from unstructured text.

Resource for Azure AI Language
1. A **Language** resource - manage access and billing for the resource separately from other services.
2. An **Azure AI services** resource - choose this resource type if you plan to use Azure AI Language in combination with other Azure AI services

-----
**Question answering with the Language Service**
You can easily create a question answering solution on Microsoft Azure using Azure AI Language service. Azure AI Language includes a custom question answering feature that enables you to create a knowledge base of question and answer pairs that can be queried using natural language input.
**Adding chit-chat to your bot makes it more conversational and engaging**. The chit-chat feature in QnA maker allows you to easily add a pre-populated set of the top chit-chat, into your knowledge base (KB). This can be a starting point for your bot's personality, and it will save you the time and cost of writing them from scratch.
You can use **Azure AI Language Studio** to create, train, publish, and manage question answering projects. You can write code to create and manage projects using the Azure AI Language REST API or SDK. However, in most scenarios it is easier to use the Language Studio.

-----
**conversational language understanding**
Azure AI Language service supports **conversational language understanding (CLU)**. You can use CLU to build language models that interpret the meaning of phrases in a conversational setting. One example of a CLU application is one that's able to turn devices on and off based on speech.
**Language Understanding (LUIS)** is a cloud-based API service that applies custom machine-learning intelligence to a user's conversational, natural language text to predict overall meaning, and pull out relevant, detailed information.
Design your LUIS model with categories of user intentions called intents. Each intent needs examples of user utterances. Each utterance can provide data that needs to be extracted with machine-learning entities.
To work with conversational language understanding (CLU), you need to take into account three core concepts: **utterances, entities, and intents**.

Resources for conversational language understanding
1. Azure AI Language: You can use a language resource for authoring and prediction.
2. Azure AI services: A general resource that includes CLU along with many other Azure AI services. You can only use this type of resource for prediction.

When you are satisfied with the results from the training and testing, you can publish your Conversational Language Understanding application to a prediction resource for consumption.

-----
https://speech.microsoft.com/

Build voice-enabled, multilingual generative AI apps with fast transcriptions and natural-sounding voices.
Customize speech in your app for your domain—including OpenAI Whisper model—or give your copilot a branded voice.
Enable real-time, multi-language speech to speech translation and speech to text transcription of audio streams.
Run AI models wherever your data resides. Deploy your apps in the cloud or at the edge with containers.

AI speech capabilities enable us to manage home and auto systems with voice instructions, get answers from computers for spoken questions, generate captions from audio, and much more.
To enable this kind of interaction, the AI system must support at least two capabilities:
1. Speech recognition - the ability to detect and interpret spoken input
2. Speech synthesis - the ability to generate spoken output
Azure AI Speech provides speech to text, text to speech, and speech translation capabilities through speech recognition and synthesis.

**Speech recognition** takes the spoken word and converts it into data that can be processed - often by transcribing it into text. 
An **acoustic model** that converts the audio signal into phonemes (representations of specific sounds). 
A **language model** that maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes.

**Speech synthesis** is concerned with vocalizing data, usually by converting text to speech. A speech synthesis solution typically requires the following information:
1. The text to be spoken
2. The voice to be used to vocalize the speech
To synthesize speech, the system typically tokenizes the text to break it down into individual words, and assigns phonetic sounds to each word. It then breaks the phonetic transcription into prosodic units (such as phrases, clauses, or sentences) to create phonemes that will be converted to audio format. These phonemes are then synthesized as audio and can be assigned a particular voice, speaking rate, pitch, and volume.

Azure AI Speech service, which supports many capabilities, including:
1. **Speech to text**: Real-time transcription & Batch transcription. Speech-to-Text, also known as automatic speech recognition (ASR), is a feature of Speech Services that provides transcription.
2. **Text to speech**: The service includes multiple pre-defined voices with support for multiple languages and regional pronunciation, including neural voices that leverage neural networks to overcome common limitations in speech synthesis with regard to intonation, resulting in a more natural sounding voice.
   
While both technologies serve to bridge the gap between humans and machines, they operate in opposite directions. **Speech synthesis takes written text and converts it into spoken language, while speech recognition interprets spoken language and converts it into text**.
A separate module covers speech translation in Azure AI services.

Azure AI Speech is available for use through several tools and programming languages including:
Studio interfaces
Command Line Interface (CLI)
REST APIs and Software Development Kits (SDKs)
You can create Azure AI Speech projects using user interfaces with Speech Studio or Azure AI Studio.

**Resources for Azure AI Speech**
A **Speech resource** - manage access and billing for the resource separately from other services.
An **Azure AI services resource** - Use Azure AI Speech in combination with other Azure AI services, and you want to manage access and billing for these services together.

-----

**Language translation concepts**

One of the many challenges of translation between languages is that words don't have a one to one replacement between languages. Machine translation advancements are needed to improve the communication of meaning and tone between languages.

**Literal and semantic translation**: A literal translation is where each word is translated to the corresponding word in the target language. Artificial intelligence systems must be able to understand, not only the words, but also the semantic context in which they're used.

**Text and speech translation**: Text translation can be used to translate documents from one language to another. Speech translation is used to translate between spoken languages, sometimes directly (speech-to-speech translation) and sometimes by translating to an intermediary text format (speech-to-text translation).

Azure AI services:
**The Azure AI Translator service**, which supports text-to-text translation. The service uses a Neural Machine Translation (NMT) model for translation
**The Azure AI Speech service**, which enables speech to text and speech-to-speech translation. return the translation as text or an audio stream.

There are dedicated Translator and Speech resource types for these services, which you can use if you want to manage access and billing for each service individually.

Text translation, Document translation & Custom translation (used to enable enterprises, app developers, and language service providers to build customized neural machine translation (NMT) systems).
The translator service provides multi-language support for text translation, transliteration, language detection, and dictionaries.

Azure AI Translator's application programming interface (API) offers some optional configuration to help you fine-tune the results that are returned, including:

1. **Profanity filtering**. Without any configuration, the service will translate the input text, without filtering out profanity. Profanity levels are typically culture-specific but you can control profanity translation by either marking the translated text as profane or by omitting it in the results.
2. **Selective translation**. You can tag content so that it isn't translated. For example, you may want to tag code, a brand name, or a word/phrase that doesn't make sense when localized.

# Document intelligence & Knowledge Mining

https://documentintelligence.ai.azure.com/studio

Azure AI Document Intelligence is the new name for **Azure Form Recognizer**. You may still see Azure Form Recognizer in the Azure portal or Document Intelligence Studio.

You can use Microsoft's **Azure AI Document Intelligence** to build solutions that manage and accelerate data collection from scanned documents. Features of Azure AI Document Intelligence help automate document processing in applications and workflows, enhance data-driven strategies, and enrich document search capabilities. The service features are available for use and testing in the **Document Intelligence Studio and other programming languages**.
**Knowledge mining** is the term used to describe solutions that involve extracting information from large volumes of often unstructured data to create a searchable knowledge store. One Microsoft knowledge mining solution is **Azure AI Search**, a private, enterprise, search solution that has tools for building indexes.

Document intelligence describes AI capabilities that support processing text and making sense of information in text. As an extension of optical character recognition (OCR), **document intelligence takes the next step** a person might after reading a form or document. It automates the process of extracting, understanding, and saving the data in text. The ability to extract text, layout, and key-value pairs is known as document analysis. Document analysis provides locations of text on a page identified by **bounding box coordinates**. 

Azure AI Document Intelligence consists of features grouped by model type:
1. **Document analysis** - general document analysis that returns structured data representations, including regions of interest and their inter-relationships.
2. **Prebuilt models** - process common document types such as invoices, business cards, ID documents, and more. financial services and legal, US tax, US mortgage, and personal identification documents.
3. **Custom models** - can be trained to identify specific fields that are not included in the existing pretrained models. **Includes custom classification models and document field extraction models** such as the custom generative AI model and custom neural model.

To use Azure AI Document Intelligence, create either a Document Intelligence or Azure AI services resource in your Azure subscription.

-----

https://learn.microsoft.com/en-us/training/modules/intro-to-azure-search/1-introduction

**Azure AI Search** provides the infrastructure and tools to create search solutions that extract data from various structured, semi-structured, and non-structured documents.

It's a Platform as a Service (PaaS) solution. Microsoft manages the infrastructure and availability, allowing your organization to benefit without the need to purchase or manage dedicated hardware resources.
Azure AI Search exists to complement existing technologies and provides a programmable search engine built on Apache Lucene, an open-source software library. 

Azure AI Search comes with the following features:

1. **Data from any source**: accepts data from any source provided in **JSON format**, with auto crawling support for selected data sources in Azure.
2. **Multiple options for search and analysis**: including vector search, full text, and hybrid search.
3. **AI enrichment**: has Azure AI capabilities built in for image and text analysis from raw content.
4. **Linguistic analysis**: offers analysis for 56 languages to intelligently handle phonetic matching or language-specific linguistics. Natural language processors available in Azure AI Search are also used by Bing and Office.
5. **Configurable user experience**: has options for query syntax including vector queries, text search, hybrid queries, fuzzy search, autocomplete, geo-search filtering based on proximity to a physical location, and more.
6. **Azure scale, security, and integration**: at the data layer, machine learning layer, and with Azure AI services and Azure OpenAI.

**A search index contains your searchable content**. In an Azure AI Search solution, you create a search index by moving data through the indexing pipeline

1. Start with a data source: the storage location of your original data artifacts, such as PDFs, video files, and images. For Azure AI Search, your data source could be files in Azure Storage, or text in a database such as Azure SQL Database or Azure Cosmos DB.
2. Indexer: automates the movement data from the data source through document cracking and enrichment to indexing. An indexer automates a portion of data ingestion and exports the original file type to JSON (in an action called JSON serialization).  An indexer serializes a source document into JSON before passing it to a search engine for indexing. An indexer automates several steps of data ingestion, reducing the amount of code you need to write.
3. Document cracking: the indexer opens files and extracts content.
4. Enrichment: the indexer moves data through AI enrichment, which implements Azure AI on your original data to extract more information. AI enrichment is achieved by adding and combining skills in a skillset. A skillset defines the operations that extract and enrich data to make it searchable. These AI skills can be either built-in skills, such as text translation or Optical Character Recognition (OCR), or custom skills that you provide. Examples of AI enrichment include adding captions to a photo and evaluating text sentiment. AI enriched content can be sent to a knowledge store, which persists output from an AI enrichment pipeline in tables and blobs in Azure Storage for independent analysis or downstream processing. A skillset requires an indexer, but an indexer doesn't require a skillset. You can use indexers to create a search index from textual content in any supported data source. **Without AI skillsets, you can still perform full text search over indexes containing alphanumeric content.** Detecting the sentiment in content requires a skillset that includes the Sentiment Analysis skill.
5. Push to index: the serialized JSON data populates the search index.
6. The result is a populated search index which can be explored through queries. When users make a search query such as "coffee", the search engine looks for that information in the search index. A search index has a structure similar to a table, known as the index schema. A typical search index schema contains fields, the field's data type (such as string), and field attributes. The fields store searchable text, and the field attributes allow for actions such as filtering and sorting. Below is an example of a search index schema:

A screenshot of the structure of an index schema in json including key phrases and image tags.

![image](https://github.com/user-attachments/assets/6b41d070-3ab9-4946-89ea-834ad9509d44)

Azure AI Search queries can be submitted as an HTTP or REST API request, with the response coming back as JSON. Queries can specify what fields are searched and returned, how search results are shaped, and how the results should be filtered or sorted. A query that doesn't specify the field to search will execute against all the searchable fields within the index.

Azure AI Search supports two types of syntax: **simple and full Lucene**. Simple syntax covers all of the common query scenarios, while full Lucene is useful for advanced scenarios.

# **Generative AI in Microsoft Azure**
**Azure OpenAI Service** is Microsoft's cloud solution for deploying, customizing, and hosting generative AI models. The service features are available for use and testing with Azure AI Foundry, Microsoft's platform for designing enterprise-grade AI solutions. You can use the **Azure AI Foundry** portal to manage, develop, and customize generative AI models.

four stage process to develop and implement a plan for responsible AI when using generative models. The four stages in the process are:

1. **Identify** potential harms that are relevant to your planned solution.
   * Identify potential harms: Generating content that is offensive, pejorative, or discriminatory, contains factual inaccuracies, encourages or supports illegal or unethical behavior or practices
   * Prioritize identified harms: The solution provides inaccurate cooking times, resulting in undercooked food that may cause illness OR the solution provides a recipe for a lethal poison that can be manufactured from everyday ingredients.
   * Test and verify the prioritized harms: Red teaming is a strategy that is often used to find security vulnerabilities or other weaknesses that can compromise the integrity of a software solution
   * Document and share the verified harms: When you have gathered evidence to support the presence of potential harms in the solution, document the details and share them with stakeholders.
2. **Measure** the presence of these harms in the outputs generated by your solution. Manual and automatic testing. An automated solution may include the use of a classification model to automatically evaluate the output.
   * Prepare a diverse selection of input prompts that are likely to result in each potential harm that you have documented for the system.
   * Submit the prompts to the system and retrieve the generated output.
   * Apply pre-defined criteria to evaluate the output and categorize it according to the level of potential harm it contains.
4. **Mitigate** the harms at multiple layers in your solution to minimize their presence and impact, and ensure transparent communication about potential risks to users. Mitigation of potential harms in a generative AI solution involves a layered approach, in which mitigation techniques can be applied at each of four layers.
   * **Model**: Selecting a model that is appropriate for the intended solution use. Fine-tuning a foundational model with your own training data so that the responses it generates are more likely to be relevant and scoped to your solution scenario.
   * **Safety System**:  Includes platform-level configurations and capabilities that help mitigate harm. For example, Azure AI Studio includes support for content filters that apply criteria to suppress prompts and responses based on classification of content into four severity levels (safe, low, medium, and high) for four categories of potential harm (hate, sexual, violence, and self-harm).
   * **Metaprompt and grounding**: focuses on the construction of prompts that are submitted to the model. 
     * Specifying metaprompts or system inputs that define behavioral parameters for the model.
     * Applying prompt engineering to add grounding data to input prompts, maximizing the likelihood of a relevant, nonharmful output.
     * Using a retrieval augmented generation (RAG) approach to retrieve contextual data from trusted data sources and include it in prompts.
   * User experience: Designing the application user interface to constrain inputs to specific subjects or types, or applying input and output validation can mitigate the risk of potentially harmful responses.
6. **Operate** the solution responsibly by defining and following a deployment and operational readiness plan. Once you identify potential harms, develop a way to measure their presence, and implement mitigations for them in your solution, you can get ready to release your solution. Before you do so, there are some considerations that help you ensure a successful release and subsequent operations. Review Legal, Privacy, Security, Accessibility. Release and operate the solution phased delivery plan, incident response plan, rollback plan etc...
   * Prompt shields: Scans for the risk of user input attacks on language models
   * Groundedness detection: Detects if text responses are grounded in a user's source content
   * Protected material detection: Scans for known copyrighted content
   * Custom categories: Define custom categories for any new or emerging patterns

# **Understand Responsible AI**
1. Fairness
2. Reliability and safety
3. Privacy and security
4. Inclusiveness
5. Transparency
6. Accountability

# **Train and understand regression models in machine learning**
Regression is a simple, common, and highly useful data analysis technique, colloquially referred to as "fitting a trend line." **Regression identifies the strength of relationship between one or more features and a single label**. In its simplest form, regression fits a straight line between a one variable (feature) and another (label). In more complicated forms, regression can find non-linear relationships between a single label and multiple features.

**Multiple linear regression** models the relationship between several features and a single variable. Mathematically, it's the same as simple linear regression, and is usually fit using the same cost function, but with more features. Rather than modeling a single relationship, this technique simultaneously models multiple relationships, **which it treats as independent of one another**. For example, if we're predicting how ill a dog becomes based on their age and body_fat_percentage, two relationships are found. The fact that the model expects features to be independent is called a **model assumption**. When model assumptions aren't true, the model can make misleading predictions.

We know that cost functions can be used to assess how well a model fits the data on which it's trained. Linear regression models have a special related measure called **R2 (R-squared)**. R2 is a value between 0 and 1 that tells us how well a linear regression model fits the data. When people talk about correlations being strong, they often mean that the R2 value was large.

R2 values are widely accepted, but aren't a perfect measure we can use in isolation. They suffer four limitations:
1. Because of how R2 is calculated, the more samples we have, the higher the R2.
2. R2 values don't tell us how well a model will work with new, previously unseen data. Statisticians overcome this by calculating a supplementary measure, called a p-value.
3. R2 values don't tell us the direction of the relationship. For example, an R2 value of 0.8 doesn't tell us whether the line is sloped upwards or downwards. It also doesn’t tell us how sloped the line is.

**Polynomial regression models** relationships as a particular type of curve. Polynomials are a family of curves, ranging from simple to complex shapes. The more parameters in the equation (model), the more complex the curve can be.

A major advantage of polynomial regression is that you can use it to look at all sorts of relationships. 
The major disadvantage to polynomial curves is that they often extrapolate poorly. In other words, if we try to predict values that are larger or smaller than our training data, polynomials can predict unrealistically extreme values. Another disadvantage is that polynomial curves are easy to overfit. This means that noise in the data can change the shape of the curve much more than simpler models, such as simple linear regression.

We've seen how multiple regression can fit several linear relationships at the same time. There's no need for these to be limited to linear relationships, though. Curves of all kinds can be used for these relationships where appropriate. Although you should take care not to use curves such as polynomials with multiple features where they're not necessary. This is because the relationships can end up very complex, which makes it harder to understand the models and assess whether they'll make predictions that don't make sense from a real-world standpoint.

# **Q&A**
You have an AI solution that provides users with the ability to control smart devices by using verbal commands. Which two types of natural language processing (NLP) workloads does the solution use? Each correct answer presents part of the solution.

Considering the scenario described where the goal is to control smart devices using voice commands, the most appropriate choice would be to use **speech-to-text conversion as the first step** in the process and **then apply language modeling to** generate consistent and meaningful responses or actions based on the commands recognized in the produced text. This could allow the AI to understand the users' intent and respond appropriately.
Key phrase extraction could also work but is more complex because an additional layer would have to be added that understands user intent based on the combination of keywords extracted. But it would become complex and probably less efficient as well. Language modeling solves this problem natively.

The metric that is not used when training models in Custom Vision is d. F1 Score. Custom Vision primarily uses **Precision, Recall, and Mean Average Precision (mAP)** to evaluate model performance

The fraction of time when the model is correct is known as: Accuracy. Accuracy is useful when the classes are balanced.
Which of these confirms how often the model is correct: Precision. Precision is useful when the cost of false positives is high.
Which value identifies how much the model finds all there is to find?: Recall. Recall is useful when the cost of false negatives is high.
F1 Score: This is the weighted average of Precision and Recall. It is useful when you need a balance between Precision and Recall, especially when the class distribution is imbalanced.

**Ordinal Regression:** This type of regression is used when the dependent variable is ordinal, meaning it has a natural order but the intervals between the values are not necessarily equal. For example, **predicting the rank of a student in a class**.
This regression type is used to predict a variable that can be considered as a label: Ordinal

**Linear Regression**: This is used to predict a continuous dependent variable based on one or more independent variables. It assumes a linear relationship between the dependent and independent variables. **For example, predicting house prices based on features like size and location**.

**Poisson Regression**: This is used for modeling count data and is appropriate when the dependent variable represents counts or the number of times an event occurs. For example, **predicting the number of customer complaints received in a day**.
This regression type uses counts instead of data values: Poisson

Support Vector Machine (SVM): While SVM is primarily used for classification tasks, it can also be adapted for regression (SVR - Support Vector Regression). It tries to find the best fit line within a certain margin of tolerance. However, it is not typically used to predict a variable that can be considered as a label.

**Statistical analysis cab be broken down into these three processes: Transformation, Visualisation, Modeling**

**Mean Absolute Error (MAE):** This is typically used in regression models to measure the average magnitude of errors between predicted and actual values. It is **not** commonly used in clustering models.

**Average Distance to Cluster Center:** This measures the average distance of all points in a cluster to the cluster center (centroid). It helps in evaluating the compactness of clusters.

**Average Distance to Other Center:** This measures the average distance of points in one cluster to the centers of other clusters. It helps in evaluating the separation between clusters.

**Number of Points:** This simply counts the number of data points in each cluster. While it can provide some information about the distribution of data points across clusters, it is not a standard metric for evaluating clustering performance.

**Tagging Images:** This involves identifying and labeling objects, scenes, and activities within an image. **Azure AI Vision**, for example, can automatically generate tags for images based on their content, making it easier to organize and search through large collections of images
