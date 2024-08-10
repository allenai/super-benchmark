
<div align="center">
    <img src="assets/mle-bench.webp" width="200"/>
    <h1 style="margin-top: 0">SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories</h1>
    <p>
    A benchmark and resources for evaluation of LLM agents on setting up and executing ML/NLP tasks from research repositories in the GitHub wild. 
    </p>
    <!--[<a href="">arxiv link here</a>]-->
</div>

---

## 📝 Benchmark Tasks

Dataset tasks are available in [HuggingFace Hub 🤗](https://huggingface.co/datasets/allenai/super).

We provide three sets: Expert (45 problems), Scenarios (152) and AutoGen (602).

Agents trajectories from the paper's experiments are available [here](trajectories).

## 🚀 Quick Start: Running the Agent

### Setup

#### 1. Clone the repo and install the requirements:
```
git clone https://github.com/allenai/super-benchmark.git
cd super-benchmark
pip install -r requirements.txt
```

#### 2. Fill in your OpenAI API key:
```
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

### Running queries

The following command will run the agent locally, which may incur risks as it will execute code on your machine.
We provide the option to run the agent inside a Docker container, and using [modal.com](https://www.modal.com/). We use the latter for the benchmark evaluation.

```bash
python -m super.run_single_query --env-backend local --query "Download the OpenBookQA dataset at https://github.com/allenai/OpenBookQA and tell me how many examples are in the train, dev, and test splits of the datasets."
```

## 🤖 Running & Evaluating Agents on SUPER

We provide code to evaluate our implemented agents on SUPER.

To run tasks safely and concurrently, we use [modal.com](https://www.modal.com/). Modal isn't free, but is quite cheap: running an average problem from the benchmark should generally cost 2-3 cents (assuming CPU).
In addition users receive $30 credit per month, which should be enough to run the benchmark evaluation multiple times. 

```bash
python -m super.run_on_benchmark --set Expert 
```