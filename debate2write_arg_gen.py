"""
This is the all-in-one module combining persona_module.py and plan_draft_module.py
All parts are separated into: 
- Persona Creation
- Discussion Modeling
- Plan Drafting
- Argument Generation
"""

import os
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from functools import lru_cache
import re
from string import punctuation
import json
import random
from loguru import logger


# Hosted open-weight model from the Hugging Face Inference Providers guide.
MODEL_INSTANCE = os.environ.get("HF_MODEL_ID", "openai/gpt-oss-120b")
MAX_LENGTH = 4000

# Original project templates are stored in prompts/ beside this script.
def _load_prompt(prompt_file):
    path = Path(__file__).resolve().parent / prompt_file
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _get_client():
    """Create a lightweight hosted inference client; never load model weights."""
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN to a Hugging Face token with Inference Providers permission."
        )
    return InferenceClient(
        api_key=token,
        provider=os.environ.get("HF_PROVIDER", "auto"),
        timeout=180,
    )


def huggingface_generate(prompt, model=MODEL_INSTANCE, max_tokens=MAX_LENGTH):
    """Call a hosted model. Install: pip install -U huggingface_hub loguru

    Configure HF_TOKEN, optionally HF_MODEL_ID and HF_PROVIDER.
    The chosen model must be served by a Hugging Face Inference Provider.
    API usage may incur charges; no local GPU or model download is required.
    """
    client = _get_client()
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The inference provider returned an empty completion.")
    return content.strip()


# Preserve imports used by earlier versions; both now call the hosted API.
openai_generate = huggingface_generate
transformers_generate = huggingface_generate


class PersonaCreator:
    def __init__(self, model=MODEL_INSTANCE, num_persona=10, max_length=4000):
        self.model = model
        self.max_length = max_length
        self.num_persona = num_persona
        self.persona_creation_prompt_file = "./prompts/persona_creation.txt"
        self.persona_selection_prompt_file = "./prompts/persona_selection.txt"
        self.persona_creation_prompt = self.load_prompt(self.persona_creation_prompt_file)
        self.persona_selection_prompt = self.load_prompt(self.persona_selection_prompt_file)

    def load_prompt(self, prompt_file):
        return _load_prompt(prompt_file)

    def generate_one(self, proposition, is_random=False, selected_num=3):
        # step 1: creation
        persona_lists = self.persona_creation(proposition)
        if not persona_lists:
            return {"persona_lists": [], "selected_persona_lists": []}
        if is_random:
            selected_persona_lists = random.sample(persona_lists, selected_num)
        else:
            selected_persona_lists = self.persona_selection(proposition, persona_lists) or []
        return {"persona_lists": persona_lists, "selected_persona_lists": selected_persona_lists}


    def persona_creation(self, proposition):
        input_prompt = self.persona_creation_prompt.replace("##input_proposition", proposition)

        output = openai_generate(input_prompt, self.model, max_tokens=self.max_length)
        
        if output is None:
            return None
        parsed_res_dict = self.extract_personas_new(output)
        logger.debug(f"persona creation: {parsed_res_dict}")
        return parsed_res_dict

    def persona_selection(self, proposition, candidate_list):
        input_prompt = self.persona_selection_prompt.replace("##input_proposition", proposition)
        candidate_text = "\n".join([json.dumps(elem) for elem in candidate_list])
        input_prompt = input_prompt.replace("###candidate_list", candidate_text)
        output = openai_generate(input_prompt, self.model, max_tokens=self.max_length)
        
        if output is None:
            return None
        parsed_res_dict = self.extract_personas_new(output)
        logger.debug(f"persona selection: {parsed_res_dict}")
    
        return parsed_res_dict

    def extract_personas_new(self, string):
        # Replace newline and multiple spaces with a single space
        string = string.strip()
        if string.startswith("```json"):
            string = string.replace("```json", "")
        if string.endswith("```"):
            string = string.replace("```", "")
        string = ' '.join(string.split())
        # Split the string into separate JSON strings
        json_strings = string.split('} {')
        # Add missing braces to each JSON string and parse into JSON objects
        json_list = []
        for s in json_strings:
            try:
                s = s.strip()
                if not s.startswith('{') and not s.endswith('}'):
                    cur_json = json.loads('{' + s + '}')
                elif not s.endswith('}'):
                    cur_json = json.loads(s + '}')
                elif not s.startswith('{'):
                    cur_json = json.loads("{" + s)
                else:
                    cur_json = json.loads(s)
                cur_descrip = cur_json["description"]
                cur_json["description"] = re.sub("Agent [0-9]{1,2}:?", "", cur_descrip).strip()
                json_list.append(cur_json)
            except: 
                logger.debug(f"{s} is not a valid JSON")
        return json_list

    def extract_personas(self, result_text):
        result_text = result_text.strip().split("\n")
        persona_list = []
        for res in result_text:
            try:
                res = res.strip().strip(",.")
                res_dict = json.loads(res)
                cur_descrip = res_dict["description"]
                res_dict["description"] = re.sub("Agent [0-9]{1,2}:?", "", cur_descrip).strip()
                persona_list.append(res_dict)
            except: 
                logger.debug(f"{res} is not a valid JSON")
       
        return persona_list


class ArgumentGenerator:
    def __init__(self, model=MODEL_INSTANCE, max_length=4000):
        self.model = model
        self.max_length = max_length
        
        self.debate_prompt_file = "./prompts/debate_discussion_noplan.txt"
        self.debate_prompt = self.load_prompt(self.debate_prompt_file)

        self.plan_prompt_file = "./prompts/plan_distillation.txt"
        self.plan_prompt = self.load_prompt(self.plan_prompt_file)

        self.surfacegen_prompt_file = "./prompts/surface_generation_step2.txt"
        self.surfacegen_prompt = self.load_prompt(self.surfacegen_prompt_file)

        self.persona_generator = PersonaCreator(model=self.model, max_length=self.max_length)


    def load_prompt(self, prompt_file):
        return _load_prompt(prompt_file)


    def generate_persona(self, proposition):
        for _ in range(3):
            persona_dict = self.persona_generator.generate_one(proposition)
            # {"persona_lists": persona_lists, "selected_persona_lists": selected_persona_lists}
            persona_lists = persona_dict["selected_persona_lists"]
            logger.debug(f"\n[log] persona_list: {persona_lists}")
            if len(persona_lists) >= 3:
                break
            logger.debug(f"[Warning Persona ERROR with {_}-th generation, will try again")
        return persona_lists


    def generate_one(self, proposition):
        result_dict = {
            "query": proposition,
        }
        ########################################################
        # Step 1: Persona Creation
        ########################################################

        persona_lists = self.generate_persona(proposition)

        result_dict["persona_lists"] = persona_lists

        assert len(persona_lists) >= 3
        # input_prompt = self.debate_prompt.replace("persona_a", json.dumps({"persona": persona_lists[0]["description"], "claim": persona_lists[0]["claim"]}).strip(".")). \
        #     replace("persona_b", json.dumps({"persona": persona_lists[1]["description"], "claim": persona_lists[1]["claim"]}).strip(".")). \
        #     replace("persona_c", json.dumps({"persona": persona_lists[2]["description"], "claim": persona_lists[2]["claim"]}).strip("."))
        # input_prompt = input_prompt.replace("{input_proposition}", proposition)
        input_prompt = self.debate_prompt.replace("persona_a", json.dumps(persona_lists[0]).strip(".")). \
            replace("persona_b", json.dumps(persona_lists[1]).strip(".")). \
            replace("persona_c", json.dumps(persona_lists[2]).strip("."))

        input_prompt = input_prompt.replace("{input_proposition}", proposition)


        ########################################################
        # Step 2: Discussion Modeling
        ########################################################
        output = openai_generate(input_prompt, self.model, max_tokens=self.max_length)
        logger.debug(f"\n[log] Discussion Modeling Output: {[output]}")
        if output is None:
            print("No answer found")
            return None  
        discussion_text = output
        result_dict["discussion"] = discussion_text

        ########################################################
        # Step 3: Plan Drafting
        ########################################################
        plan_input_prompt = self.plan_prompt.format(input_proposition=proposition, discussion_process=discussion_text)
        for label, persona in zip(("persona_a", "persona_b", "persona_c"), persona_lists):
            plan_input_prompt = plan_input_prompt.replace(label, json.dumps(persona))
        output = openai_generate(plan_input_prompt, self.model, max_tokens=self.max_length)
        logger.debug(f"\n[log] Plan Drafting Output: {[output]}")
        if output is None:
            return None  
        plan_text = output.replace("\n\n", "\n")

        result_dict["plan"] = plan_text

        ########################################################
        # Step 4: Argument Generation
        ########################################################
        surface_gen_input = self.surfacegen_prompt.format(proposition=proposition, plan=plan_text)
        output = openai_generate(surface_gen_input, self.model, max_tokens=self.max_length)
        logger.debug(f"\n[log] Argument Generation Output: {[output]}")
        if output is None:
            return None  
        argument_text = output

        result_dict["argument"] = argument_text

        return result_dict

    def extract_components(self, result_text):
        # old template
        # self.discussion_pattern = r"Discussion Process:([\s\S]*)Final Plan"
        # self.plan_pattent = r"Final Plan:([\s\S]*)"

        self.discussion_pattern = r"start_of_discussion([\s\S]*)end_of_discussion"
        self.plan_pattent = r"_start_of_plan([\s\S]*)end_of_plan"

        discussion_matches = re.findall(self.discussion_pattern, result_text)
        plan_matches = re.findall(self.plan_pattent, result_text)

        if len(discussion_matches) == 0:
            discussion_str = None
        else:
            discussion_str = discussion_matches[0].strip(punctuation).strip()
        
        if len(plan_matches) == 0:
            plan_str = None
        else:
            plan_str = plan_matches[0].strip(punctuation).strip()   

        res_dict = {"discussion": discussion_str, "plan": plan_str}
        return res_dict


def save_result(result, output_dir=None):
    """Save the complete workflow and a readable argument without overwriting runs."""
    if result is None:
        raise ValueError("No result was generated; nothing to save.")
    directory = (
        Path(output_dir) if output_dir is not None
        else Path(__file__).resolve().parent / "results"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]
    json_path = directory / f"argument_{stamp}.json"
    text_path = directory / f"argument_{stamp}.txt"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with text_path.open("x", encoding="utf-8") as handle:
        handle.write(f"Proposition: {result['query']}\n\n{result['argument']}\n")
    return json_path.resolve(), text_path.resolve()


if __name__ == "__main__":
    arg_generator = ArgumentGenerator(model=MODEL_INSTANCE)

    statement = "We should allow more economic permits for foreign workers in the EU."
    print(statement)
    result = arg_generator.generate_one(statement)
    print(result)
    if result is not None:
        json_path, text_path = save_result(result)
        print(f"Full result saved to: {json_path}")
        print(f"Final argument saved to: {text_path}")
    else:
        print("No result was generated; no files were saved.")
