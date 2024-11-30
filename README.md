# Logprob-Free Attack on Language Models

This repository contains an implementation of a logprob-free attack on pre-trained language models. The approach allows extraction of model logits without explicitly querying for the log-probabilities of tokens, instead using binary search and logit manipulation methods.

## Features
- Extract logits using a basic logprob-free attack.
- Approximate hidden layer weights with Singular Value Decomposition (SVD).
- Support for multiple pre-trained language models (e.g., GPT-Neo).

## Requirements

To install the dependencies, see the `requirements.txt` file.

## Installation

1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script to perform logit extraction:
    ```bash
    python logprob_free_attack.py
    ```

2. Modify the script to use different prompts or token ranges, as per your requirements.

### Explanation of Functions
- `get_Q()`: Extracts logits for a set of prompts and constructs matrix Q.
- `dim_extraction()`: Estimates the hidden dimension of the model using SVD on matrix Q.
- `layer_extraction()`: Reconstructs approximate layer weights from the SVD output.
- `binary_search_extraction()`: Performs a binary search to determine the logit value for a target token.

## Example

To estimate the hidden layer dimension and recover logits:
- First, the pre-trained GPT-Neo model is loaded.
- Then, logit extraction is performed using prompts, followed by dimensionality extraction via SVD.

## License
[MIT](https://opensource.org/licenses/MIT)

## Acknowledgments
This project is inspired by the paper "Stealing Part of a Production Language Model." Special thanks to the authors and community contributors who have provided the foundational resources.
