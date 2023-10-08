import autogen
import transformers
import logging
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def generate_requirements():
    # Generate requirements.txt file from Pipfile
    os.system("pipenv lock -r > requirements.txt")

def get_config_list():
    # Get the config list for all three models
    config_list_gpt4 = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["microsoft/DialoGPT-small", "TheBloke/Mistral-7B-OpenOrca-GGUF", "TheBloke/Llama-2-70B-chat-GPTQ"],
        },
    )
    return config_list_gpt4

def create_config_list(config_list_gpt4):
    # Create a list of Hugging Face config dictionaries
    config_list = [] # Initialize an empty list

    for config in config_list_gpt4:
      # Remove the unnecessary config parameters for the other two models
      if "api_base" in config:
        config.pop("api_base")
      if "api_type" in config:
        config.pop("api_type")
      if "api_version" in config:
        config.pop("api_version")

      # Append a dictionary to the list
      config_list.append({
        "model": config["model"],
        "api_key": "<your HF_KEY API key here>",
        "seed": 42,
        "temperature": 0.9,
        "config_list": [
            {
                "min_length": 10,
                "max_length": 100,
                "do_sample": True,
                "num_beams": 5,
                "early_stopping": True
            }
        ],
        "request_timeout": 120
      })

    return config_list

def load_dialogpt_config():
    # Load the config file for the microsoft/DialoGPT-small model
    with open("dialogpt_config.json") as f:
        dialogpt_config = json.load(f)
    
    return dialogpt_config

def define_roles_and_responsibilities():
    # Define the roles and responsibilities of the assistant agents
    assistant_agents = {
        "devops manager": "you're responsible for overall success of the project...",
        "devops analyst": "you're responsible for collecting, analyzing, reporting on data...",
        "devops developer": "you for developing, testing, and deploying the project's applications...",
        "devops security analyst": "you are responsible for identifying and mitigating security risks...",
        "devops automation engineer": "you're responsible for automating tasks and processes..."
    }

    return assistant_agents

def create_assistant_agents(assistant_agents, dialogpt_config):
    # Create the assistant agents
    for agent_name, agent_role in assistant_agents.items():
        try:
            # Use autogen.AssistantAgent() function (documented in autogen module)
            agent = autogen.AssistantAgent(
                name=agent_name,
                llm_config=dialogpt_config, # Use the dialogpt_config variable here
                system_message=agent_role,
            )
            logging.info(f"Created assistant agent {agent_name}")
        except Exception as e:
            logging.error(f"Failed to create assistant agent {agent_name}: {e}")
            raise e

    return assistant_agents

def create_group_chat(assistant_agents):
    # Define the list of agents for the group chat
    agents = [agent for agent in assistant_agents.values()]

    # Create a group chat object using the autogen.GroupChat class
    groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=50)

    return groupchat

def assign_leader(assistant_agents):
    # Assign the DevOps Manager agent to lead the group
    assistant_agents["devops manager"].is_leader = True

def create_manager(groupchat, dialogpt_config):
    # Create a manager object using the autogen.GroupChatManager class
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=dialogpt_config)

    return manager

def create_user_proxy():
    # Create a user proxy agent using the autogen.UserProxyAgent class
    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A human user. Interact with the group chat and provide feedback.",
        human_input_mode="ALWAYS",
        code_execution_config=False,
    )

    return user_proxy

def initiate_chat(user_proxy, manager):
    # Initiate a chat using the user_proxy.initiate_chat method
    user_proxy.initiate_chat(
        manager,
        message="Hello everyone. I'm here to monitor your progress on the project."
    )

def main():
    load_model()
    generate_requirements()
    config_list_gpt4 = get_config_list()
    config_list = create_config_list(config_list_gpt4)
    dialogpt_config = load_dialogpt_config()
    assistant_agents = define_roles_and_responsibilities()
    assistant_agents = create_assistant_agents(assistant_agents, dialogpt_config)
    groupchat = create_group_chat(assistant_agents)
    assign_leader(assistant_agents)
    manager = create_manager(groupchat, dialogpt_config)
    user_proxy = create_user_proxy()
    initiate_chat(user_proxy, manager)

if __name__ == "__main__":
    main()
```
