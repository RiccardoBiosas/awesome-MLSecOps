[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/RiccardoBiosas/awesome-MLSecOps/graphs/commit-activity)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![GitHub](https://img.shields.io/twitter/follow/axsaucedo.svg?label=Follow)](https://twitter.com/RBiosas)
# Awesome MLSecOps

A curated list of awesome open-source tools, resources, and tutorials for MLSecOps (Machine Learning Security Operations).

[![mm.png](https://i.postimg.cc/VL6C6tKF/mm.png)](https://postimg.cc/SjBs1nFX)

## Table of Contents

- [Open Source Security Tools](#open-source-security-tools)
- [ML code security](#ml-code-security)
- [101](#101)
- [Attack Vectors](#attack-vectors)
- [Blogs and Publications](#blogs-and-publications)
- [MlOps infrastracture vulnerabilities](#mlops-infrastracture-vulnerabilities)
- [Community Resources](#community-resources)
- [Infographics](#infographics)
- [Contributions](#contributions)


## Open Source Security Tools
- [ModelScan](https://github.com/protectai/modelscan) - Protection Against ML Model Serialization Attacks.
- [NB Defense](https://nbdefense.ai) - Secure Jupyter Notebooks.
- [Garak](https://github.com/leondz/garak) -  LLM vulnerability scanner.
- [Adversarial Robustness Toolbox](https://github.com/IBM/adversarial-robustness-toolbox) - A library of defense methods for machine learning models against adversarial attacks.
- [MLSploit](https://github.com/mlsploit/) - MLsploit is a cloud framework for interactive experimentation with adversarial machine learning research.
- [TensorFlow Privacy](https://github.com/tensorflow/privacy) - A library of privacy-preserving machine learning algorithms and tools.
- [Foolbox](https://github.com/bethgelab/foolbox) - A Python toolbox for creating and evaluating adversarial attacks and defenses.
- [Advertorch](https://github.com/BorealisAI/advertorch) - A Python toolbox for adversarial robustness research. 
- [Artificial Intelligence Threat Matrix](https://collaborativeaicontrols.github.io/ATM/) - A framework for identifying and mitigating threats to machine learning systems.
- [Adversarial ML Threat Matrix](https://github.com/mitre/advmlthreatmatrix) - Adversarial Threat Landscape for AI Systems.
- [CleverHans](https://github.com/cleverhans-lab/cleverhans) - A library of adversarial examples and defenses for machine learning models.
- [AdvBox](https://github.com/advboxes/AdvBox) - Advbox is a toolbox to generate adversarial examples that fool neural networks in PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow.
- [Audit AI](https://github.com/pymetrics/audit-ai) - Bias Testing for Generalized Machine Learning Applications.
- [Deep Pwning](https://github.com/cchio/deep-pwning) - Deep-pwning is a lightweight framework for experimenting with machine learning models with the goal of evaluating their robustness against a motivated adversary. 
- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) - An open-source library to audit data privacy in statistical and machine learning algorithms.
- [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) - A library for analyzing, validating, and monitoring machine learning models in production.
- [PromptInject](https://github.com/agencyenterprise/PromptInject) - A framework that assembles adversarial prompts.
- [TextAttack](https://github.com/QData/TextAttack) - TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP.
- [OpenAttack](https://github.com/thunlp/OpenAttack) - An Open-Source Package for Textual Adversarial Attack.
- [TextFooler](https://github.com/jind11/TextFooler) - A Model for Natural Language Attack on Text Classification and Inference.
- [Flawed Machine Learning Security](https://github.com/EthicalML/fml-security) - Practical examples of "Flawed Machine Learning Security" together with ML Security best practice across the end to end stages of the machine learning model lifecycle from training, to packaging, to deployment.
- [Adversarial Machine Learning CTF](https://github.com/arturmiller/adversarial_ml_ctf) - This repository is a CTF challenge, showing a security flaw in most (all?) common artificial neural networks. They are vulnerable for adversarial images.
- [Damn Vulnerable LLM Project](https://github.com/harishsg993010/DamnVulnerableLLMProject) - A Large Language Model designed for getting hacked
- [Gandalf Lakera](https://gandalf.lakera.ai/) - Prompt Injection CTF playground
- [Vigil](https://github.com/deadbits/vigil-llm) - LLM prompt injection and security scanner
- [PALLMs (Payloads for Attacking Large Language Models)](https://github.com/mik0w/pallms) - list of various payloads for attacking LLMs collected in one place
- [AI-exploits](https://github.com/protectai/ai-exploits) - exploits for MlOps systems. It's not just in the inputs given to LLMs such as ChatGPT
- [Offensive ML Playbook](https://wiki.offsecml.com/Welcome+to+the+Offensive+ML+Playbook) - Offensive ML Playbook. Notes on machine learning attacks and pentesting.
- [AnonLLM](https://github.com/fsndzomga/anonLLM) - Anonymize Personally Identifiable Information (PII) for Large Language Model APIs.
- [AI Goat](https://github.com/dhammon/ai-goat) - vulnerable LLM CTF challenges.
- [Pyrit](https://github.com/Azure/PyRIT) - The Python Risk Identification Tool for generative AI.
- [Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors](https://github.com/advmlphish/raze_to_the_ground_aisec23) - Source code of the paper "Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors" accepted at AISec '23


## ML code security
- [lintML](https://github.com/JosephTLucas/lintML) - security linter for ML, by Nvidia
- [HiddenLayer: Model as Code](https://hiddenlayer.com/research/models-are-code/) - research about some vectors in ml libraries.
- [Copycat CNN](https://github.com/jeiks/Stealing_DL_Models) - proof-of-concept on how to generate a copy of a Convolutional Neural Network by querying it as a black-box with random data and using the output to train a copycat CNN which mimics the target CNN's predictive patterns.

## 101
- [AI Security 101](https://www.nightfall.ai/ai-security-101)
- [Web LLM attacks](https://portswigger.net/web-security/llm-attacks)
- [Microsoft AI Red Team](https://learn.microsoft.com/en-us/security/ai-red-team/)
- [AI Risk Assessment for ML Engineers](https://learn.microsoft.com/en-us/security/ai-red-team/ai-risk-assessment)
- 
  
## Attack Vectors
- [Data Poisoning](https://github.com/ch-shin/awesome-data-poisoning)
- [Adversarial Prompt Exploits](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks)
- [Evasion Attack](https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/)
- [Membership Inference Exploits](https://arxiv.org/abs/2103.07853)

## Blogs and Publications 
- [Red-Teaming Large Language Models](https://huggingface.co/blog/red-teaming)
- [Google's AI red-team](https://blog.google/technology/safety-security/googles-ai-red-team-the-ethical-hackers-making-ai-safer/)
- [The MLSecOps Top 10 vulnerabilities](https://ethical.institute/security.html)
- [Token Smuggling Jailbreak via Adversarial Prompt](https://www.piratewires.com/p/gpt4-token-smuggling)
- [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and
Data Poisoning Attacks](https://arxiv.org/pdf/2006.12557.pdf)
- [We need a new way to measure AI security](https://blog.trailofbits.com/2023/03/14/ai-security-safety-audit-assurance-heidy-khlaaf-odd/)
- [PrivacyRaven: Implementing a proof of concept for model inversion](https://blog.trailofbits.com/2021/11/09/privacyraven-implementing-a-proof-of-concept-for-model-inversion/)
- [Adversarial Prompts Engineering](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-adversarial.md)
- [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909)
- [Trail Of Bits' audit of Hugging Face's safetensors library](https://github.com/trailofbits/publications/blob/master/reviews/2023-03-eleutherai-huggingface-safetensors-securityreview.pdf)
- [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/)
- [LLM Security](https://llmsecurity.net/)
- [Is you MLOps infrastructure leaking secrets?](https://hackstery.com/2023/10/13/no-one-is-prefect-is-your-mlops-infrastructure-leaking-secrets/)
- [Embrace The Red, blog where show how u can hack LLM's.](https://embracethered.com/)
- [Audio-jacking: Using generative AI to distort live audio transactions](https://securityintelligence.com/posts/using-generative-ai-distort-live-audio-transactions/)
- [HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/)
- 

## MlOps infrastracture vulnerabilities
- [SILENT SABOTAGE](https://hiddenlayer.com/research/silent-sabotage/) - A study on bot compromise for converting Pickle to SafeTensors.
- [NOT SO CLEAR: HOW MLOPS SOLUTIONS CAN MUDDY THE WATERS OF YOUR SUPPLY CHAIN](https://hiddenlayer.com/research/not-so-clear-how-mlops-solutions-can-muddy-the-waters-of-your-supply-chain/) - This study examines vulnerabilities for the ClearML platform.
- [Uncovering Azure's Silent Threats: A Journey into Cloud Vulnerabilities](https://www.youtube.com/watch?v=tv8tei97Sv8) - This study shows the security issues of Azure MLAAS(Machine Learning As A Service).
- [The MlOps Security Landscape](https://hackstery.com/wp-content/uploads/2023/11/mlops_owasp_oslo_2023.pdf)
- [Confused Learning: Supply Chain Attacks through Machine Learning Models](https://blackhat.com/asia-24/briefings/schedule/#confused-learning-supply-chain-attacks-through-machine-learning-models-37794) - Released in April 2024.

## Community Resources

- [MLSecOps](https://mlsecops.com/)
- [MLSecOps Podcast](https://mlsecops.com/podcast)
- [MITRE ATLAS™ (Adversarial Threat Landscape for Artificial-Intelligence Systems)](https://atlas.mitre.org/)
- [OWASP Machine Learning Security Top Ten](https://owasp.org/www-project-machine-learning-security-top-10/)
- [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security)
- [Hackstery](https://hackstery.com/)
- [PWNAI](https://t.me/pwnai)
- [AiSec_X_Feed](https://t.me/aisecnews)
- [HUNTR Discord community](https://discord.com/invite/GBmmty82CM)
- [AIRSK](https://airisk.io)
- [AI Vulnerability Database](https://avidml.org/)
- [Incident AI Database](https://incidentdatabase.ai/)
- [Defcon AI Villiage CTF](https://www.kaggle.com/competitions/ai-village-ctf/overview)
- [Awesome AI Security](https://github.com/ottosulin/awesome-ai-security)
- [MLSecOps Reference Repository](https://github.com/disesdi/mlsecops_references)


## Infographics

### MlSecOps lifecycle
[![MLSecOps_Lifecycle](https://github.com/RiccardoBiosas/awesome-MLSecOps/assets/65150720/236cd3f4-bce8-4659-b43f-8d4002df65a5)](https://www.conf42.com/DevSecOps_2022_Eugene_Neelou_ai_introducing_mlsecops_for_software_20)


### Ai Security Market map
[![Market Map](https://i.postimg.cc/15ZxH0q9/marketmap.png)](https://menlovc.com/perspective/security-for-ai-genai-risks-and-the-emerging-startup-landscape/)

## Contributions
All contributions to this list are welcome!


## Contributors ✨
- [<img src='https://github.com/riccardobiosas.png?size=50'>](https://github.com/riccardobiosas) [@riccardobiosas](https://github.com/riccardobiosas)
- [<img src='https://github.com/badarahmed.png?size=50'>](https://github.com/badarahmed) [@badarahmed](https://github.com/badarahmed)
- [<img src='https://github.com/deadbits.png?size=50'>](https://github.com/deadbits) [@deadbits](https://github.com/deadbits)
- [<img src='https://github.com/wearetyomsmnv.png?size=50'>](https://github.com/wearetyomsmnv) [@wearetyomsmnv](https://github.com/wearetyomsmnv)
- [<img src='https://github.com/anmorgan24.png?size=50'>](https://github.com/anmorgan24) [@anmorgan24](https://github.com/anmorgan24)
- [<img src='https://github.com/mik0w.png?size=50'>](https://github.com/mik0w) [@mik0w](https://github.com/mik0w)


