#!/usr/bin/env python3
"""
Prompt Expansion Script

This script uses Gemini Flash 2.5 to probabilistically modify prompts from prompts.txt
based on attribute distributions defined in prompt-expansion.json. The script dynamically
reads all attribute keys from the configuration file, making it flexible to work with
any attributes (ethnicity, gender, age, profession, etc.).
"""

import json
import random
import argparse
import sys
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Tuple


def load_config(config_path: str) -> Dict:
    """Load the prompt expansion configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}")
        sys.exit(1)


def load_prompts(prompts_path: str) -> List[str]:
    """Load prompts from the text file."""
    try:
        with open(prompts_path, 'r') as f:
            # Read all lines and strip whitespace, filter out empty lines
            prompts = [line.strip() for line in f.readlines() if line.strip()]
            return prompts
    except FileNotFoundError:
        print(f"Error: Prompts file {prompts_path} not found.")
        sys.exit(1)


def sample_from_distribution(distribution: Dict[str, float]) -> str:
    """Sample a value from a probability distribution."""
    items = list(distribution.items())
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    
    return random.choices(values, weights=weights, k=1)[0]


def expand_prompt_with_gemini(prompt: str, sampled_attributes: Dict[str, str], 
                            template: str, model) -> str:
    """Use Gemini to expand a single prompt with the specified attributes."""
    try:
        # Format the template with the provided values
        # Add the prompt as a special key
        format_dict = sampled_attributes.copy()
        format_dict['prompt'] = prompt
        
        formatted_prompt = template.format(**format_dict)
        
        # Generate response using Gemini
        response = model.generate_content(formatted_prompt)
        
        if response.text:
            return response.text.strip()
        else:
            print(f"Warning: Empty response from Gemini for prompt: {prompt}")
            return prompt  # Return original prompt if expansion fails
            
    except Exception as e:
        print(f"Error expanding prompt with Gemini: {e}")
        return prompt  # Return original prompt if expansion fails


def main():
    parser = argparse.ArgumentParser(description='Expand prompts using Gemini Flash 2.5 with configurable attributes')
    parser.add_argument('api_key', help='Google AI API key for Gemini')
    parser.add_argument('--config', default='prompt-expansion.json', 
                       help='Path to configuration file (default: prompt-expansion.json)')
    parser.add_argument('--input', default='data/prompts.txt',
                       help='Path to input prompts file (default: data/prompts.txt)')
    parser.add_argument('--output', default='expanded-prompts.txt',
                       help='Path to output file (default: expanded-prompts.txt)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Load prompts
    print(f"Loading prompts from {args.input}...")
    prompts = load_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize Gemini
    print("Initializing Gemini Flash 2.5...")
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Get distributions and template
    distributions = config['distributions']
    template = config['prompt']
    
    # Print all distributions
    for attr_name, dist in distributions.items():
        print(f"{attr_name.title()} distribution: {dist}")
    
    # Process each prompt
    expanded_prompts = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
        
        # Sample all attributes dynamically
        sampled_attributes = {}
        for attr_name, dist in distributions.items():
            sampled_value = sample_from_distribution(dist)
            sampled_attributes[attr_name] = sampled_value
        
        # Print sampled values
        sampled_str = ", ".join([f"{k}: {v}" for k, v in sampled_attributes.items()])
        print(f"  Sampled: {sampled_str}")
        
        # Expand prompt using Gemini
        expanded_prompt = expand_prompt_with_gemini(
            prompt, sampled_attributes, template, model
        )
        
        expanded_prompts.append(expanded_prompt)
        print(f"  Expanded: {expanded_prompt[:80]}...")
        print()
    
    # Save expanded prompts (append mode)
    print(f"Appending expanded prompts to {args.output}...")
    with open(args.output, 'a') as f:
        for prompt in expanded_prompts:
            f.write(prompt + '\n')
    
    print(f"Successfully expanded {len(expanded_prompts)} prompts!")
    print(f"Results appended to {args.output}")


if __name__ == "__main__":
    main()
