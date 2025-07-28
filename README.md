# Awesome MLSecOps 🛡️🤖

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/RiccardoBiosas/awesome-MLSecOps/graphs/commit-activity)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![Follow](https://img.shields.io/twitter/follow/RBiosas.svg?style=social&label=Follow)](https://twitter.com/RBiosas)

A curated list of awesome open-source tools, resources, and tutorials for MLSecOps (Machine Learning Security Operations).

![MLSecOps Banner](https://github.com/user-attachments/assets/966affca-442a-4859-b450-774b8d48c6cc)


## Table of Contents
- [Open Source Security Tools](#open-source-security-tools)
- [Commercial Tools](#commercial-tools)
- [DATA](#data)
- [ML Code Security](#ml-code-security)
- [101 Resources](#101-resources)
- [Attack Vectors](#attack-vectors)
- [Blogs and Publications](#blogs-and-publications)
- [MLOps Infrastructure Vulnerabilities](#mlops-infrastructure-vulnerabilities)
- [Community Resources](#community-resources)
- [Infographics](#infographics)
- [Contributions](#contributions)
- [Contributors](#contributors-)


## Open Source Security Tools

#### In this section, you and I can take a look at what opensource solutions and PoCs, exist to accomplish the task of ML protection. Of course, some of them are unsupported or will have difficulties to run. However, not mentioning them is a big crime.

| Tool | Description |
|------|-------------|
| [ModelScan](https://github.com/protectai/modelscan) | Protection Against ML Model Serialization Attacks |
| [NB Defense](https://nbdefense.ai) | Secure Jupyter Notebooks |
| [Garak](https://github.com/leondz/garak) | LLM vulnerability scanner |
| [Adversarial Robustness Toolbox](https://github.com/IBM/adversarial-robustness-toolbox) | Library of defense methods for ML models against adversarial attacks |
| [MLSploit](https://github.com/mlsploit/) | Cloud framework for interactive experimentation with adversarial machine learning research |
| [TensorFlow Privacy](https://github.com/tensorflow/privacy) | Library of privacy-preserving machine learning algorithms and tools |
| [Foolbox](https://github.com/bethgelab/foolbox) | Python toolbox for creating and evaluating adversarial attacks and defenses |
| [Advertorch](https://github.com/BorealisAI/advertorch) | Python toolbox for adversarial robustness research |
| [Artificial Intelligence Threat Matrix](https://collaborativeaicontrols.github.io/ATM/) | Framework for identifying and mitigating threats to machine learning systems |
| [Adversarial ML Threat Matrix](https://github.com/mitre/advmlthreatmatrix) | Adversarial Threat Landscape for AI Systems |
| [CleverHans](https://github.com/cleverhans-lab/cleverhans) | A library of adversarial examples and defenses for machine learning models|
| [AdvBox](https://github.com/advboxes/AdvBox) | Advbox is a toolbox to generate adversarial examples that fool neural networks in PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow|
| [Audit AI](https://github.com/pymetrics/audit-ai) | Bias Testing for Generalized Machine Learning Applications|
| [Deep Pwning](https://github.com/cchio/deep-pwning) | Deep-pwning is a lightweight framework for experimenting with machine learning models with the goal of evaluating their robustness against a motivated adversary|
| [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) | An open-source library to audit data privacy in statistical and machine learning algorithms|
| [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) | A library for analyzing, validating, and monitoring machine learning models in production|
| [PromptInject](https://github.com/agencyenterprise/PromptInject) | A framework that assembles adversarial prompts|
| [TextAttack](https://github.com/QData/TextAttack) | TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP|
| [OpenAttack](https://github.com/thunlp/OpenAttack) | An Open-Source Package for Textual Adversarial Attack|
| [TextFooler](https://github.com/jind11/TextFooler) | A Model for Natural Language Attack on Text Classification and Inference|
| [Flawed Machine Learning Security](https://github.com/EthicalML/fml-security) | Practical examples of "Flawed Machine Learning Security" together with ML Security best practice across the end to end stages of the machine learning model lifecycle from training, to packaging, to deployment|
| [Adversarial Machine Learning CTF](https://github.com/arturmiller/adversarial_ml_ctf) | This repository is a CTF challenge, showing a security flaw in most (all?) common artificial neural networks. They are vulnerable for adversarial images|
| [Damn Vulnerable LLM Project](https://github.com/harishsg993010/DamnVulnerableLLMProject) | A Large Language Model designed for getting hacked|
| [Gandalf Lakera](https://gandalf.lakera.ai/) | Prompt Injection CTF playground|
| [Vigil](https://github.com/deadbits/vigil-llm) | LLM prompt injection and security scanner|
| [PALLMs (Payloads for Attacking Large Language Models)](https://github.com/mik0w/pallms) | list of various payloads for attacking LLMs collected in one place|
| [AI-exploits](https://github.com/protectai/ai-exploits) | exploits for MlOps systems. It's not just in the inputs given to LLMs such as ChatGPT|
| [Offensive ML Playbook](https://wiki.offsecml.com/Welcome+to+the+Offensive+ML+Playbook) | Offensive ML Playbook. Notes on machine learning attacks and pentesting|
| [AnonLLM](https://github.com/fsndzomga/anonLLM) | Anonymize Personally Identifiable Information (PII) for Large Language Model APIs|
| [AI Goat](https://github.com/dhammon/ai-goat) | vulnerable LLM CTF challenges|
| [Pyrit](https://github.com/Azure/PyRIT) | The Python Risk Identification Tool for generative AI|
| [Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors](https://github.com/advmlphish/raze_to_the_ground_aisec23) | Source code of the paper "Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors" accepted at AISec '23|
| [Giskard](https://github.com/Giskard-AI/giskard) | Open-source testing tool for LLM applications|
| [Safetensors](https://github.com/huggingface/safetensors) | Convert pickle to a safe serialization option|
| [Citadel Lens](https://www.citadel.co.jp/en/products/lens/)| Quality testing of models according to industry standards|
| [Model-Inversion-Attack-ToolBox](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox) | A framework for implementing Model Inversion attacks|
| [NeMo-Guardials](https://github.com/NVIDIA/NeMo-Guardrails) | NeMo Guardrails allow developers building LLM-based applications to easily add programmable guardrails between the application code and the LLM|
| [AugLy](https://github.com/facebookresearch/AugLy) | A tool for generating adversarial attacks|
| [Knockoffnets](https://github.com/tribhuvanesh/knockoffnets) | PoC to implement BlackBox attacks to steal model data|
| [Robust Intelligence Continous Validation](https://www.robustintelligence.com/platform/continuous-validation) | Tool for continuous model validation for compliance with standards|
| [VGER](https://github.com/JosephTLucas/vger) | Jupyter Attack framework |
| [AIShield Watchtower](https://github.com/bosch-aisecurity-aishield/watchtower) | An open source tool from AIShield for studying AI models and scanning for vulnerabilities|
| [PS-fuzz](https://github.com/prompt-security/ps-fuzz) | tool for scanning LLM vulnerabilities|
| [Mindgard-cli](https://github.com/Mindgard/cli/) | Check security of you AI via CLI|
| [PurpleLLama3](https://meta-llama.github.io/PurpleLlama/)| Check LLM security with Meta LLM Benchmark |
| [Model transparency](https://github.com/sigstore/model-transparency) | generate model signing |
| [ARTkit](https://github.com/BCG-X-Official/artkit)|Automated prompt-based testing and evaluation of Gen AI applications|
| [LangBiTe](https://github.com/SOM-Research/LangBiTe) | A Bias Tester framework for LLMs|
| [OpenDP](https://github.com/opendp/opendp)| The core library of differential privacy algorithms powering the OpenDP Project|
| [TF-encrypted](https://tf-encrypted.io/)| Encryption for tensorflow|
| [Agentic Security](https://github.com/msoedov/agentic_security)| Agentic LLM Vulnerability Scanner / AI red teaming kit|
| [Promptfoo Scanner](https://github.com/promptfoo/promptfoo) | An open-source LLM red teaming tool |


## Commercial Tools

| Tool | Description |
|------|-------------|
| [Databricks Platform, Azure Databricks](https://azure.microsoft.com/ru-ru/products/databricks) | Datalake data management and implementation tool |
| [Hidden Layer AI Detection Response](https://hiddenlayer.com/aidr/) | Tool for detecting and responding to incidents |
| [Guardian](https://protectai.com/guardian) | Model protection in CI/CD |
| [Promptfoo](https://www.promptfoo.dev/security/) | Continuous monitoring, detection, and remediation for enterprise LLM applications |

## DATA

| Tool | Description |
|------|-------------|
| [ARX - Data Anonymization Tool](https://arx.deidentifier.org/) | Tool for anonymizing datasets |
| [Data-Veil](https://veil.ai/) | Data masking and anonymization tool |
| [Tool for IMG anonymization](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/blob/main/ch10/notebooks/Image%20Anonymization.ipynb)| Image anonymization|
| [Tool for DATA anonymization](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/blob/main/ch10/notebooks/Data%20Anonymization.ipynb)| Data anonymization|
| [BMW-Anonymization-Api](https://github.com/BMW-InnovationLab/BMW-Anonymization-API?referral=top-free-anonymization-tools-apis-and-open-source-models)| This repository allows you to anonymize sensitive information in images/videos. The solution is fully compatible with the DL-based training/inference solutions that we already published/will publish for Object Detection and Semantic Segmentation |
| [DeepPrivacy2](https://github.com/hukkelas/deep_privacy2)| A Toolbox for Realistic Image Anonymization |
| [PPAP](https://github.com/tgisaturday/PPAP)|Latent-space-level Image Anonymization with Adversarial Protector Networks|

## ML Code Security

- [lintML](https://github.com/JosephTLucas/lintML) - Security linter for ML, by Nvidia
- [HiddenLayer: Model as Code](https://hiddenlayer.com/research/models-are-code/) - Research about some vectors in ML libraries
- [Copycat CNN](https://github.com/jeiks/Stealing_DL_Models) - Proof-of-concept on how to generate a copy of a Convolutional Neural Network
- [differential-privacy-library](https://github.com/IBM/differential-privacy-library) - Library designed for differential privacy and machine learning

## 101 Resources

#### You can find here a list of resources to help you get into the topic of AI security. Understand what attacks exist and how they can be used by an attacker.

- [AI Security 101](https://www.nightfall.ai/ai-security-101)
- [Web LLM attacks](https://portswigger.net/web-security/llm-attacks)
- [Microsoft AI Red Team](https://learn.microsoft.com/en-us/security/ai-red-team/)
- [AI Risk Assessment for ML Engineers](https://learn.microsoft.com/en-us/security/ai-red-team/ai-risk-assessment)
- [Microsoft - Generative AI Security for beginners](https://github.com/microsoft/generative-ai-for-beginners/blob/main/13-securing-ai-applications/README.md?WT.mc_id=academic-105485-koreyst)

#### AI Security Study Map

[![AI Security Study Map](https://i.postimg.cc/G2QdqnK6/map.png)](https://postimg.cc/sQvkg8tJ)

[Full size map in this repository](https://github.com/wearetyomsmnv/AI-LLM-ML_security_study_map)

## Threat Modeling

- [AI Villiage: LLM threat modeling](https://aivillage.org/large%20language%20models/threat-modeling-llm/)

- [JSOTIRO/ThreatModels](https://github.com/jsotiro/ThreatModels/tree/main)

![image](https://github.com/user-attachments/assets/367a50da-c93d-4c91-a69f-9a6de8d48f91)

![image](https://github.com/user-attachments/assets/eae84861-945b-4a2e-8037-f5ccfc92e5d0)

![image](https://github.com/user-attachments/assets/9f366c92-3e5a-4375-b967-ac35801151c1)

![image](https://github.com/user-attachments/assets/db78c3e7-8e41-4097-8f71-30b69eb70e55)

![image](https://github.com/user-attachments/assets/2cc30071-7ec2-4f09-bf80-29d6b1a008ba)


more in **Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps**


## Attack Vectors

#### Here we provide a useful list of resources that focus on a specific attack vector.

- [Data Poisoning](https://github.com/ch-shin/awesome-data-poisoning)
- [Adversarial Prompt Exploits](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks)
- [Model Inversion Attacks](https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/)
- [Model Evasion Attacks](https://www.ibm.com/docs/en/watsonx/saas?topic=atlas-evasion-attack)
- [Membership Inference Exploits](https://arxiv.org/abs/2103.07853)
- [Model Stealing Attacks](https://arxiv.org/abs/2206.08451)
- [ML Supply Chain Attacks](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML06_2023-AI_Supply_Chain_Attacks)
- [Model Denial Of Service](https://genai.owasp.org/llmrisk/llm04-model-denial-of-service/)
- [Gradient Leakage Attacks](https://ieeexplore.ieee.org/document/10107713)
- [Cloud Infrastructure Attacks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7931962/)
- [Architecture and Access-control attacks](https://genai.owasp.org/llmrisk/llm08-excessive-agency/)


## Blogs and Publications 

#### 🌱 The AI security community is growing. New blogs and many researchers are emerging. In this paragraph you can see examples of some blogs.

- 📚 [What is MLSecOps](https://themlsecopshacker.com/p/what-is-mlsecops)
- 🛡️ [Red-Teaming Large Language Models](https://huggingface.co/blog/red-teaming)
- 🔍 [Google's AI red-team](https://blog.google/technology/safety-security/googles-ai-red-team-the-ethical-hackers-making-ai-safer/)
- 🔒 [The MLSecOps Top 10 vulnerabilities](https://ethical.institute/security.html)
- 🏴‍☠️ [Token Smuggling Jailbreak via Adversarial Prompt](https://www.piratewires.com/p/gpt4-token-smuggling)
- ☣️ [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and
Data Poisoning Attacks](https://arxiv.org/pdf/2006.12557.pdf)
- 📊 [We need a new way to measure AI security](https://blog.trailofbits.com/2023/03/14/ai-security-safety-audit-assurance-heidy-khlaaf-odd/)
- 🕵️ [PrivacyRaven: Implementing a proof of concept for model inversion](https://blog.trailofbits.com/2021/11/09/privacyraven-implementing-a-proof-of-concept-for-model-inversion/)
- 🧠 [Adversarial Prompts Engineering](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-adversarial.md)
- 🔫 [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909)
- 📋 [Trail Of Bits' audit of Hugging Face's safetensors library](https://github.com/trailofbits/publications/blob/master/reviews/2023-03-eleutherai-huggingface-safetensors-securityreview.pdf)
- 🔝 [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/)
- 🔐 [LLM Security](https://llmsecurity.net/)
- 🔑 [Is you MLOps infrastructure leaking secrets?](https://hackstery.com/2023/10/13/no-one-is-prefect-is-your-mlops-infrastructure-leaking-secrets/)
- 🚩 [Embrace The Red, blog where show how u can hack LLM's.](https://embracethered.com/)
- 🎙️ [Audio-jacking: Using generative AI to distort live audio transactions](https://securityintelligence.com/posts/using-generative-ai-distort-live-audio-transactions/)
- 🌐 [HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/)
- 🧰 [WTF-blog - MlSecOps frameworks ... Which ones are available and what is the difference?](https://blog.wearetyomsmnv.wtf/articles/mlsecops-frameworks-...-which-ones-are-available-and-what-is-the-difference)
- 📚 [DreadNode Paper Stack](https://dreadnode.notion.site/2582fe5306274c60b85a5e37cf99da7e?v=74ab79ed1452441dab8a1fa02099fed)

## MLOps Infrastructure Vulnerabilities

#### Very interesting articles on MlOps infrastructure vulnerabilities. In some of them you can even find ready-made exploits.

- [SILENT SABOTAGE](https://hiddenlayer.com/research/silent-sabotage/) - Study on bot compromise for converting Pickle to SafeTensors
- [NOT SO CLEAR: HOW MLOPS SOLUTIONS CAN MUDDY THE WATERS OF YOUR SUPPLY CHAIN](https://hiddenlayer.com/research/not-so-clear-how-mlops-solutions-can-muddy-the-waters-of-your-supply-chain/) - Study on vulnerabilities for the ClearML platform
- [Uncovering Azure's Silent Threats: A Journey into Cloud Vulnerabilities](https://www.youtube.com/watch?v=tv8tei97Sv8) - Study on security issues of Azure MLAAS
- [The MLOps Security Landscape](https://hackstery.com/wp-content/uploads/2023/11/mlops_owasp_oslo_2023.pdf)
- [Confused Learning: Supply Chain Attacks through Machine Learning Models](https://blackhat.com/asia-24/briefings/schedule/#confused-learning-supply-chain-attacks-through-machine-learning-models-37794) 

## MlSecOps pipeline

![image](https://github.com/user-attachments/assets/8ce8400b-804b-4ce0-9241-30ad5b42b55f)



# Academic Po(C)ker FACE

## Repositories

#### [AgentPoison](https://github.com/BillChan226/AgentPoison)
Official implementation of "AgentPoison: Red-teaming LLM Agents via Memory or Knowledge Base Backdoor Poisoning". This project explores methods of data poisoning and backdoor insertion in LLM agents to assess their resilience against such attacks.

#### [DeepPayload](https://github.com/yuanchun-li/DeepPayload)
Research on methods of embedding malicious payloads into deep neural networks.

#### [backdoor](https://github.com/bolunwang/backdoor)
Investigation of backdoor attacks on deep learning models, focusing on creating undetectable vulnerabilities within models.

#### [Stealing_DL_Models](https://github.com/jeiks/Stealing_DL_Models)
Techniques for stealing deep learning models through various attack vectors, enabling adversaries to replicate or access models.

#### [datafree-model-extraction](https://github.com/cake-lab/datafree-model-extraction)
Model extraction without using data, allowing for the recovery of models without access to the original data.

#### [LLMmap](https://github.com/pasquini-dario/LLMmap)
Tool for mapping and analyzing large language models (LLMs), exploring the structure and behavior of various LLMs.

#### [GoogleCloud-Federated-ML-Pipeline](https://github.com/raj200501/GoogleCloud-Federated-ML-Pipeline)
Federated learning pipeline using Google Cloud infrastructure, enabling model training on distributed data.

#### [Class_Activation_Mapping_Ensemble_Attack](https://github.com/DreamyRainforest/Class_Activation_Mapping_Ensemble_Attack)
Attack using ensemble class activation maps to introduce errors in models by manipulating activation maps.

#### [COLD-Attack](https://github.com/Yu-Fangxu/COLD-Attack)
Methods for attacking deep models under various conditions and constraints, focusing on creating more resilient attacks.

#### [pal](https://github.com/chawins/pal)
Research on adaptive attacks on machine learning models, enabling the creation of attacks that can adapt to model defenses.

#### [ZeroShotKnowledgeTransfer](https://github.com/polo5/ZeroShotKnowledgeTransfer)
Knowledge transfer in zero-shot scenarios, exploring methods to transfer knowledge between models without prior training on target data.

#### [GMI-Attack](https://github.com/AI-secure/GMI-Attack)
Attack for generating informative labels, aimed at covertly extracting data from trained models.

#### [Knowledge-Enriched-DMI](https://github.com/SCccc21/Knowledge-Enriched-DMI)
Enhancing DMI (Data Mining and Integration) methods using additional knowledge to improve accuracy and efficiency.

#### [vmi](https://github.com/wangkua1/vmi)
Research on methods for visualizing and interpreting machine learning models, providing insights into model workings.

#### [Plug-and-Play-Attacks](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
Attacks that can be "plugged and played" without needing model modifications, offering flexible and universal attack methods.

#### [snap-sp23](https://github.com/johnmath/snap-sp23)
Tool for analyzing and processing snapshot data, enabling efficient handling of data snapshots.

#### [privacy-vs-robustness](https://github.com/inspire-group/privacy-vs-robustness)
Research on the trade-offs between privacy and robustness in models, aiming to balance these two aspects in machine learning.

#### [ML-Leaks](https://github.com/AhmedSalem2/ML-Leaks)
Methods for data leakage from trained models, exploring ways to extract private information from machine learning models.

#### [BlindMI](https://github.com/hyhmia/BlindMI)
Research on blind information extraction attacks, enabling data retrieval without access to the model's internal structure.

#### [python-DP-DL](https://github.com/NNToan-apcs/python-DP-DL)
Differential privacy methods for deep learning, ensuring data privacy during model training.

#### [MMD-mixup-Defense](https://github.com/colaalex111/MMD-mixup-Defense)
Defense methods using MMD-mixup, aimed at improving model robustness against attacks.

#### [MemGuard](https://github.com/jinyuan-jia/MemGuard)
Tools for protecting memory from attacks, exploring ways to prevent data leaks from model memory.

#### [unsplit](https://github.com/ege-erdogan/unsplit)
Methods for merging and splitting data to improve training, optimizing the use of heterogeneous data in models.

#### [face_attribute_attack](https://github.com/koushiksrivats/face_attribute_attack)
Attacks on face recognition models using attributes, exploring ways to manipulate facial attributes to induce errors.

#### [FVB](https://github.com/Sanjana-Sarda/FVB)
Attacks on face verification models, aimed at disrupting authentication systems based on face recognition.

#### [Malware-GAN](https://github.com/yanminglai/Malware-GAN)
Using GANs to create malware, exploring methods for generating malicious code with generative models.

#### [Generative_Adversarial_Perturbations](https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations)
Methods for generating adversarial perturbations using generative models, aimed at introducing errors in deep models.

#### [Adversarial-Attacks-with-Relativistic-AdvGAN](https://github.com/GiorgosKarantonis/Adversarial-Attacks-with-Relativistic-AdvGAN)
Adversarial attacks using Relativistic AdvGAN, exploring methods for creating more realistic and effective attacks.

#### [llm-attacks](https://github.com/llm-attacks/llm-attacks)
Attacks on large language models, exploring vulnerabilities and protection methods for LLMs.

#### [LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)
Safe fine-tuning of large language models, aiming to prevent data leaks and ensure security during LLM tuning.

#### [DecodingTrust](https://github.com/AI-secure/DecodingTrust)
Methods for evaluating trust in models, exploring ways to determine the reliability and safety of machine learning models.

#### [promptbench](https://github.com/microsoft/promptbench)
Benchmark for evaluating prompts, providing tools for testing and optimizing queries to large language models.

#### [rome](https://github.com/kmeng01/rome)
Tool for analyzing and evaluating models based on ROM codes, exploring various aspects of model performance and resilience.

#### [llmprivacy](https://github.com/eth-sri/llmprivacy)
Research on privacy in large language models, aiming to protect data and prevent leaks from LLMs.





## Community Resources

- [MLSecOps](https://mlsecops.com/)
- [MLSecOps Podcast](https://mlsecops.com/podcast)
- [MITRE ATLAS™](https://atlas.mitre.org/) and [SLACK COMMUNITY](https://join.slack.com/t/mitreatlas/shared_invite/zt-10i6ka9xw-~dc70mXWrlbN9dfFNKyyzQ)
- [MlSecOps communtiy](https://mlsceops.com) and [SLACK COMMUNITY](https://mlsecops.slack.com/)
- [MITRE ATLAS™ (Adversarial Threat Landscape for Artificial-Intelligence Systems)](https://atlas.mitre.org/)
- [OWASP AI Exchange](https://owaspai.org)
- [OWASP Machine Learning Security Top Ten](https://owasp.org/www-project-machine-learning-security-top-10/)
- [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP LLMSVS](https://owasp.org/www-project-llm-verification-standard/)
- [OWASP Periodic Table of AI Security](https://owaspai.org/goto/periodictable/)
- [OWASP SLACK](https://owasp.org/slack/invite)
- [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security)
- [Hackstery](https://hackstery.com/)
- [PWNAI](https://t.me/pwnai)
- [AiSec_X_Feed](https://t.me/aisecnews)
- [HUNTR Discord community](https://discord.com/invite/GBmmty82CM)
- [AIRSK](https://airisk.io)
- [AI Vulnerability Database](https://avidml.org/)
- [Incident AI Database](https://incidentdatabase.ai/)
- [Defcon AI Village CTF](https://www.kaggle.com/competitions/ai-village-ctf/overview)
- [Awesome AI Security](https://github.com/ottosulin/awesome-ai-security)
- [MLSecOps Reference Repository](https://github.com/disesdi/mlsecops_references)
- [Awesome LVLM Attack](https://github.com/liudaizong/Awesome-LVLM-Attack)
- [Awesome MLLM Safety](https://github.com/isXinLiu/Awesome-MLLM-Safety)


## Books

- [Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps](https://www.amazon.com/Adversarial-Attacks-Mitigations-Defense-Strategies/dp/1835087981)
- [Privacy-Preserving Machine Learning](https://www.ebooks.com/en-cg/book/211334202/privacy-preserving-machine-learning/srinivasa-rao-aravilli/)
- [Generative AI Security: Theories and Practices (Future of Business and Finance) ](https://www.amazon.com/Generative-AI-Security-Theories-Practices/dp/3031542517)

## Infographics

### MLSecOps Lifecycle
[![MLSecOps Lifecycle](https://github.com/RiccardoBiosas/awesome-MLSecOps/assets/65150720/236cd3f4-bce8-4659-b43f-8d4002df65a5)](https://www.conf42.com/DevSecOps_2022_Eugene_Neelou_ai_introducing_mlsecops_for_software_20)

### AI Security Market Map
[![Market Map](https://i.postimg.cc/15ZxH0q9/marketmap.png)](https://menlovc.com/perspective/security-for-ai-genai-risks-and-the-emerging-startup-landscape/)

## Contributions

All contributions to this list are welcome! Please feel free to submit a pull request with any additions or improvements.

## Contributors ✨

<table>
  <tr>
    <td align="center"><a href="https://github.com/riccardobiosas"><img src="https://github.com/riccardobiosas.png" width="100px;" alt=""/><br /><sub><b>@riccardobiosas</b></sub></a></td>
    <td align="center"><a href="https://github.com/badarahmed"><img src="https://github.com/badarahmed.png" width="100px;" alt=""/><br /><sub><b>@badarahmed</b></sub></a></td>
    <td align="center"><a href="https://github.com/deadbits"><img src="https://github.com/deadbits.png" width="100px;" alt=""/><br /><sub><b>@deadbits</b></sub></a></td>
    <td align="center"><a href="https://github.com/wearetyomsmnv"><img src="https://github.com/wearetyomsmnv.png" width="100px;" alt=""/><br /><sub><b>@wearetyomsmnv</b></sub></a></td>
    <td align="center"><a href="https://github.com/anmorgan24"><img src="https://github.com/anmorgan24.png" width="100px;" alt=""/><br /><sub><b>@anmorgan24</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mik0w"><img src="https://github.com/mik0w.png" width="100px;" alt=""/><br /><sub><b>@mik0w</b></sub></a></td>
    <td align="center"><a href="https://github.com/alexcombessie"><img src="https://github.com/alexcombessie.png" width="100px;" alt=""/><br /><sub><b>@alexcombessie</b></sub></a></td>
  </tr>
</table>


## Repository Stats

![GitHub stars](https://img.shields.io/github/stars/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub forks](https://img.shields.io/github/forks/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/RiccardoBiosas/awesome-MLSecOps)
![GitHub issues](https://img.shields.io/github/issues/RiccardoBiosas/awesome-MLSecOps)
![GitHub pull requests](https://img.shields.io/github/issues-pr/RiccardoBiosas/awesome-MLSecOps)

## Activity

![Repo activity](https://img.shields.io/github/commit-activity/m/RiccardoBiosas/awesome-MLSecOps)
![Contributors](https://img.shields.io/github/contributors/RiccardoBiosas/awesome-MLSecOps)

## Support Us

If you find this project useful, please consider giving it a star ⭐️

[![GitHub Sponsor](https://img.shields.io/github/sponsors/RiccardoBiosas?style=social)](https://github.com/sponsors/RiccardoBiosas)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

---

<p align="center">Made with ❤️</p>




