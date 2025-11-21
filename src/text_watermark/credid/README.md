# CredID: Credible LLM Watermarking Framework and Toolkit

## Overview

This repository contains the open-source toolkit and research materials for the "[Credibility-Driven Multi-bit Watermark for Large Language Models Identification](https://arxiv.org/abs/2412.03107)" project. The project introduces a novel multi-party credible watermarking framework that enhances the identification, privacy, and security of Large Language Models (LLMs). The watermarking scheme is designed to embed and extract multi-bit information without compromising the quality of the generated text. 

## Key Features

- **Multi-party Participation**: A collaborative watermarking framework involving a Trusted Third Party (TTP) and multiple LLM vendors.
- **Privacy Protection**: Ensures that user prompts and model outputs remain confidential during the watermarking process.
- **Flexibility and Scalability**: Easily accommodates different vendors' private watermarking technologies and allows for the upgrading of authentication and encryption algorithms.
- **Robust Watermarking Algorithm**: A multi-bit watermarking method that improves success rates and information capacity, with a focus on maintaining text quality.
- **Open-source Toolkit**: A user-friendly toolkit that includes watermarking frameworks, experiment configurations, datasets, and baseline algorithms.

## Repository Structure

```
llm-watermarking/
│
├── paper/                  # Research paper and related documents
│   ├── Credibility-Driven_Multi-bit_Watermark.pdf
│   └── Reproducibility_Checklist.md
│
├── config/                 # Configuration files for various watermark algorithms          
│   ├── CredID.json       
│   ├── EXP.json           
│   ├── KGW.json
│   ├── CTWL.json            
│   ├── SIR.json            
│   ├── SWEET.json         
│   ├── Unigram.json        
│   ├── MPAC.json           
│   └── Cyclic-shift.json    
│       
├── src/                    # Source code for the watermarking framework
│   ├── embedding/          # Embedding watermark into LLM responses
│   ├── extraction/         # Extracting watermark from text
│   └── utils/              # Utility functions and modules
│
├── datasets/               # Sample datasets used in the experiments
│   ├── multi_vendors_data/
│   ├── OpenGen/
│   ├── C4_News/
│   └── Essays/
│
├── watermarking/          #Implementation framework for watermark algorithms
│   ├── watermark_processor.py
│   ├── base.py
│   ├── KGW/
│   ├── EXP/
│   ├── SIR/
│   ├── SWEET/
│   ├── Unigram/
│   ├── CredID/
│   ├── CTWL/
│   ├── MPAC/
│   └── Cyclic-shift/
│
├── attacks/                #Implementation framework for watermark algorithms
│   ├── get_datasets.py
│   ├── cp_attack.py
│   ├── Deletion_attack.py/
│   ├── Substitution_attack.py
│   └── Homoglyph_attack.py
│
├── evaluation/                 # Evaluation module, including tools and pipelines
│   ├── dataset.py              # Script for handling dataset operations within evaluations
│   ├── examples/               # Scripts for automated evaluations using pipelines
│   │   ├── assess_success_rate.py  
│   │   ├── assess_quality.py
│   │   ├── assess_speed.py    
│   │   └── assess_robustness.py   
│   ├── pipelines/              # Pipelines for structured evaluation processes
│   │   ├── speed_analysis.py    
│   │   ├── success_rate_analysis.py    
│   │   ├── multi_voting_analysis.py  
│   │   └── quality_analysis.py 
│   └── tools/                  # Evaluation tools
│       ├── ppl.py
│       ├── metrics/ 
│       └── BLEU.py   
│
├── demo/                   # Credible watermark framework cases and examples for user testing
│   ├── test_method_single_party.py      
│   ├── test_pipeline_multi_party.py    
│   └── test_real_word.py
│
├── experiments/            # Scripts and configurations for running experiments
│
├── README.md               # Main project documentation
└── requirements.txt        # Dependencies required for the project
```

## Installation

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository:

   ```
   git clone https://anonymous.4open.science/r/credible-LLM-watermarking-7D62.git
   ```

2. Navigate to the repository:

   ```
   cd credible-LLM-watermarking
   ```

3. Create a virtual environment and activate it (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

## Quick Start

The toolkit provides functionalities for both watermark embedding and extraction. Here are the basic steps to use the toolkit:

### Embedding Watermarks

1. Prepare your LLM and the text prompt, and converts model version information, timestamps, and extra information into a binary sequence of a specific length as a secret message.

   watermark message generation:

   ```python
   model_version = "LLaMA2-7Bv1.2.3"
   timestamps = "202408011234"
   extra_info = True
   length = 10
   # generate secret message
    message = secret_message_gen(model_version, timestamps, extra_info, length)
   # message:[0011110001]
   ```
2. Configure the watermark parameters, and write the message into the `message` field in the JSON：

   ```
   #./config/CredID.json
    {
       "method": "CredID",
       "prompt": "A new study has found that the universe is made of",
       "model_name": "huggyllama/llama-7b",
       "oracle_model_name": "huggyllama/llama-13b",
       "dataset_path": "./datasets/c4-train.00000-of-00512_sliced",
       "log_file": "example.log",
       "sample_idx": 77,
       "max_new_tokens": 110,
       "num_beams": 4,
       "min_length": 10000,
       "penalty": 1.5,
       "lm_message_model_params": {
           "delta": 1.5,
           "lm_prefix_len": 10,
           "lm_topk": -1,
           "message_code_len": 10,
           "random_permutation_num": 50
       },
       "wm_processor_message_model_params": {
           "encode_ratio": 10,
           "max_confidence_lbd": 0.5,
           "strategy": "max_confidence_updated",
           "message": [0011110001]
       }
   }
   ```



3. Use the embedding module to integrate the watermark into the LLM's response.

   ``````python
   output_tokens = model.generate(
            **tokenized_input, 
            max_new_tokens=config['max_new_tokens'], 
            num_beams=config['num_beams'],
            logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model]))
   output_text = tokenizer.decode(output_tokens[0][start_length:], skip_special_tokens=True)
   print("watermarked_text:",output_text)
   >>> 4.9% ordinary matter, 26.8% dark matter and 68.3% dark energy.Scientists have unveiled a comprehensive overview of what makes up our Universe. Astronomers at Durham University, led by Professor Carlos Frenk, used data from NASA's Wilkinson Microwave Anisotropy Probe (WMAP) to create the first-ever 'cosmic census'. Their findings are published in the journal Monthly Notices
   ``````

### Extracting Watermarks

1. Obtain the watermarked text.  
2. Utilize the extraction module to decode and verify the watermark.

   ```python
   log_probs = wm_precessor_message_model.decode(output_text)
   print("my watermark confidence:", log_probs[1][1][0][2])
   >>> my watermark confidence:0.99872
   print("extracted message:", log_probs[0])
   >>> extracted message:[0011110001]
   ```

### Running Demo

To run the demo, follow these steps:

1. Navigate to the demo directory:
```
cd ./demo
```
2. For a basic usage scenario that simulates a single model vendor using our CredID for watermark embedding and extraction, we provide a demo that includes the following modules:

- **Model Loading**: Load the pre-trained model required for watermark embedding and extraction.
- **Watermark Parameter Loading**: Load the parameters necessary for the watermarking process.
- **Watermark Embedding Pipeline**: Embed the watermark into the text using the loaded model and parameters.
- **Watermark Extraction Pipeline**: Extract and verify the watermark from the text to ensure its integrity.
- **Text Quality Analysis**: Analyze the quality of the text after watermark embedding to ensure it meets the required standards.

To run the demo, execute the following command:
```
python test_method_single_party.py
```
3. For a scenario where multiple model vendors use our CredID for watermark embedding and extraction, we provide a demo that uses three models with corresponding watermark parameters and watermark messages. The generated watermarked texts are then fed into the watermark extraction pipelines of all three models for joint-voting extraction.

To run the demo, execute the following command:
```
python test_method_multi_party.py
```
4. To further simulate a real-world internet scenario where there is a mix of human-written non-watermarked text and multi-vendor watermarked text, we provide a demo that uses a text mixture dataset we constructed. This demo utilizes the joint-voting extraction framework to identify the source of each type of text.

To run the demo, execute the following command:
```
python test_real_world.py
```
The data for this test is located in `./datasets/multi_vendors_data`.

### Running Experiments

1. Configure the experiment settings and run the experiment scripts in the `experiments/` directory.
   
   - `analysis_CTWL_CredID.py`
     - This script performs analysis on the CTWL or CredID methods. It includes statistical evaluations and case studies to understand the method characteristics.
   - `analysis_KGW.py`
     - This script performs analysis on the KGW family methods. It includes statistical evaluations and case studies to understand the method characteristics.
   - `experiment.py`
     -  A general-purpose script that sets up and runs various experiments. It include multiple configurations and can be customized for different experimental setups.
   - `run_CredID.py`
      - This script is specifically designed to run experiments on CredID framework. It includes data loading, model training, and evaluation procedures.
   - `run_attack.py`
      -  This script simulates and analyzes attacks on watermarked texts. It is used to evaluate the robustness and security of the watermarking against specific types of attacks.
   - `wm_test_CredID_single_party.py`
     -  This script tests CredID watermarking in a single-party setting. It evaluates the success rates and quality of the watermarking method.
   -  `wm_test_ecc_7_4.py`
        - This script is used to test CredID equipped with error-correcting codes (ECC) with parameters (7, 4). It includes encoding, decoding, and performance evaluation components.
   -  `wm_test_realword.py`
      -  This script tests watermarking techniques in real-world scenarios. It evaluates how well the watermarking perform under practical conditions.
   - `wm_test_rs.py`
     - This script tests random sampling (RS) techniques for watermarking. It includes procedures for sampling, watermark embedding, and evaluation.

2. Run the watermark attack scripts in the `attacks/` directory.
   
   
   - `cp_attack.py`
     - This script performs a specific type of copy-paste attack. It includes methods to automate the attack process and measure the robustness of the watermarking.
   
   - `deletion.py`
     - This script simulates deletion attacks where parts of the watermarked text are removed. It assesses how well the watermarking method can withstand and recover from such attacks.
   
   - `homoglyphs_attack.py`
     - This script simulates homoglyph attacks, where visually similar characters are substituted in the watermarked text. It evaluates the robustness of the watermarking method against these subtle alterations.
   
   - `substitution_attack.py`
     - This script simulates substitution attacks, where certain characters or words in the watermarked text are replaced. It evaluates the robustness of the watermarking method to such modifications.


3. Run the evaluation scripts to evaluate the watermarking performance in the `evaluation/pipelines/` directory.

   - `multi_voting_analysis.py`
     - This script performs analysis using multi-voting techniques. It evaluates the effectiveness of watermarking methods based on aggregated voting results from multiple sources.
   
   - `quality_analysis.py`
     - This script assesses the quality of the watermarked texts. It includes metrics and evaluations to determine the perceptual quality and integrity of the watermarked content.
   
   - `speed_analysis.py`
     - This script analyzes the speed performance of the watermarking methods. It measures the time taken for watermark embedding and detection processes under different conditions.
   
   - `success_rate_analysis.py`
     - This script evaluates the success rate of watermark extraction. It includes procedures to calculate and analyze the success rates under various scenarios and attack conditions.




## Contributing

We welcome contributions to the project. Please follow the guidelines in `CONTRIBUTING.md` for submitting issues and pull requests.

## License

This project is licensed under the [LICENSE_NAME](LICENSE). Please refer to the license file for more information.

## Citation

If you use this toolkit in your research, please cite our paper:

```bibtex
@misc{jiang2024credidcrediblemultibitwatermark,
      title={CredID: Credible Multi-Bit Watermark for Large Language Models Identification}, 
      author={Haoyu Jiang and Xuhong Wang and Ping Yi and Shanzhe Lei and Yilun Lin},
      year={2024},
      eprint={2412.03107},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.03107}, 
}
```

## Acknowledgements

We would like to acknowledge the contributions of the LLM research community and the developers of the open-source models used in our experiments. This repository contains code for the paper [Towards Codable Watermarking for Large Language Models](https://github.com/lancopku/codable-watermarking-for-llm) by Lean Wang, Wenkai Yang, Deli Chen, Hao Zhou, Yankai Lin, Fandong Meng, Jie Zhou and Xu Sun, and the paper [MarkLLM: An Open-source toolkit for LLM Watermarking](https://github.com/THU-BPM/MarkLLM) by Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, Irwin King and Philip S. Yu.

