from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import os

def create_log_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def parse_input_data(args: tuple, kwargs: dict) -> str:
    """Convert input arguments to markdown format."""
    md_content = "## Input\n\n### Arguments\n"
    
    if args:
        md_content += "\nPositional arguments:\n"
        for i, arg in enumerate(args):
            md_content += f"- Arg {i}: `{arg}`\n"
    else:
        md_content += "\nNo positional arguments\n"
    
    md_content += "\n### Keyword Arguments\n"
    if kwargs:
        for key, value in kwargs.items():
            if key == "messages":
                md_content += f"\n#### Messages:\n"
                for msg in value:
                    md_content += f"- Role: `{msg.get('role', 'N/A')}`\n"
                    md_content += f"  Content: `{msg.get('content', 'N/A')}`\n"
            else:
                md_content += f"- {key}: `{value}`\n"
    else:
        md_content += "\nNo keyword arguments\n"
    
    return md_content

def parse_output_data(response: Any) -> str:
    """Convert response data to markdown format."""
    md_content = "\n## Output\n\n"
    
    if hasattr(response, "model"):
        md_content += f"- Model: `{response.model}`\n"
    
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            md_content += f"- Completion: `{choice.message.content}`\n"
    
    if hasattr(response, "usage"):
        md_content += "\n### Usage Statistics\n"
        md_content += f"- Completion Tokens: `{response.usage.completion_tokens}`\n"
        md_content += f"- Prompt Tokens: `{response.usage.prompt_tokens}`\n"
        md_content += f"- Total Tokens: `{response.usage.total_tokens}`\n"
    
    return md_content

def save_parsed_log(input_data: str, output_data: str, base_dir: Path):
    """Save log data as markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    md_content = f"# LLM Interaction Log - {timestamp}\n\n"
    md_content += input_data
    md_content += output_data
    
    log_file = base_dir / f"log_{timestamp}.md"
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(md_content)

def add_parser_to_provider(provider_class):
    """Decorator to add parsing functionality to the LiteLLMProvider class."""
    original_log_interaction = provider_class._log_interaction
    
    def new_log_interaction(self, kwargs: dict, response: Any):
        # Create log directory if it doesn't exist
        logs_dir = create_log_directory()
        
        # Parse input and output data to markdown
        input_md = parse_input_data((), kwargs)
        output_md = parse_output_data(response)
        
        # Save markdown log
        save_parsed_log(input_md, output_md, logs_dir)
        
        # Call original method to maintain existing functionality
        return original_log_interaction(self, kwargs, response)
    
    provider_class._log_interaction = new_log_interaction
    return provider_class