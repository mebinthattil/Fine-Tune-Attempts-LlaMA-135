"""
This script is AI generated and not intended to use used as a standalone inference script, its just a quick script for this use case. Use at your own caution.
Chat-style inference script for LlaMA-135M fine-tuned model.
Maintains conversation history and formats inputs according to training data structure.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import argparse
import sys
import warnings
import os

# Try to import llama-cpp-python for GGUF support
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# Try to import huggingface_hub for downloading GGUF models
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ConversationAwareChatbot:
    def __init__(self, model_name: str, max_context_tokens: int = 512, mode: int = 1):
        """
        Initialize the chatbot with the fine-tuned model.
        Supports both HuggingFace models and GGUF models.
        
        Args:
            model_name: HuggingFace model name or local path to GGUF file
            max_context_tokens: Maximum tokens for context (conservative for 135M model)
            mode: Generation mode (1=default, 2=low_temp, 3=deterministic)
        """
        print(f"Loading model: {model_name}")
        print(f"Generation mode: {mode}")
        
        # Define generation parameters based on mode
        self.generation_settings = self._get_mode_settings(mode)
        
        # Detect if this is a GGUF model
        self.is_gguf = self._is_gguf_model(model_name)
        self.max_context_tokens = max_context_tokens
        self.conversation_history = []
        
        if self.is_gguf:
            self._init_gguf_model(model_name)
        else:
            self._init_hf_model(model_name)
        
        print(f"Model loaded successfully!")
        print(f"Model type: {'GGUF' if self.is_gguf else 'HuggingFace'}")
        print(f"Device: {'CPU' if self.is_gguf else ('CUDA' if torch.cuda.is_available() else 'CPU')}")
        print(f"Max context tokens: {max_context_tokens}")
        print(f"Temperature: {self.generation_settings['temperature']}, do_sample: {self.generation_settings['do_sample']}")
    
    def _get_mode_settings(self, mode: int) -> dict:
        """Get generation settings based on mode."""
        base_settings = {
            "max_tokens": 150,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        
        if mode == 1:
            # Mode 1: Default (temp=0.7, do_sample=True)
            base_settings.update({
                "temperature": 0.7,
                "do_sample": True,
            })
        elif mode == 2:
            # Mode 2: Low temperature (temp=0.3, do_sample=True) 
            base_settings.update({
                "temperature": 0.3,
                "do_sample": True,
            })
        elif mode == 3:
            # Mode 3: Higher temp with deterministic (temp=0.7, do_sample=False)
            base_settings.update({
                "temperature": 0.7,
                "do_sample": False,
            })
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 1, 2, or 3.")
            
        return base_settings
    
    def _is_gguf_model(self, model_name: str) -> bool:
        """Check if the model is a GGUF model."""
        # Check if it's a local file with .gguf extension
        if os.path.isfile(model_name) and model_name.lower().endswith('.gguf'):
            return True
        
        # Check if it's a HuggingFace model name that contains GGUF indicators
        gguf_indicators = ['gguf', 'GGUF', 'ggml', 'GGML']
        return any(indicator in model_name for indicator in gguf_indicators)
    
    def _init_gguf_model(self, model_name: str):
        """Initialize GGUF model using llama-cpp-python."""
        if not GGUF_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install it with: pip install llama-cpp-python"
            )
        
        # Handle different model sources
        model_path = self._get_gguf_model_path(model_name)
        
        print(f"Loading GGUF model from: {model_path}")
        
        # Initialize GGUF model
        self.model = Llama(
            model_path=model_path,
            n_ctx=self.max_context_tokens,
            n_threads=4,  # Adjust based on your CPU
            verbose=False
        )
        
        # Store the actual model path for metadata
        self.model_path = model_path
        
        # GGUF models don't need separate tokenizer
        self.tokenizer = None
        
        # Generation parameters for GGUF - use centralized settings
        self.generation_params = {
            "max_tokens": self.generation_settings["max_tokens"],
            "temperature": self.generation_settings["temperature"],
            "top_p": self.generation_settings["top_p"],
            "top_k": self.generation_settings["top_k"],
            "repeat_penalty": self.generation_settings["repetition_penalty"],
            "stop": ["Student:"]  # GGUF-specific stop sequences
        }
    
    def _get_gguf_model_path(self, model_name: str) -> str:
        """Get the local path for a GGUF model, downloading if necessary."""
        # If it's already a local file, return as-is
        if os.path.isfile(model_name):
            return model_name
        
        # If it's a HuggingFace model, try to download it
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface-hub is required for downloading GGUF models from HuggingFace. "
                "Install it with: pip install huggingface-hub"
            )
        
        try:
            # Try common GGUF filenames
            possible_filenames = [
                "model.gguf",
                "ggml-model.gguf", 
                "pytorch_model.gguf",
                f"{model_name.split('/')[-1]}.gguf"
            ]
            
            print(f"Searching for GGUF file in HuggingFace repository: {model_name}")
            
            # Try to download the GGUF file
            for filename in possible_filenames:
                try:
                    print(f"  Trying filename: {filename}")
                    model_path = hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        cache_dir=None  # Use default cache
                    )
                    print(f"  Found and downloaded: {filename}")
                    return model_path
                except Exception as e:
                    print(f"  Failed to download {filename}: {str(e)[:100]}...")
                    continue
            
            # If none of the common names work, list repository files
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_name)
                gguf_files = [f for f in files if f.lower().endswith('.gguf')]
                
                if gguf_files:
                    filename = gguf_files[0]  # Use the first GGUF file found
                    print(f"  Found GGUF file: {filename}")
                    model_path = hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        cache_dir=None
                    )
                    return model_path
                else:
                    raise FileNotFoundError(f"No GGUF files found in repository {model_name}")
                    
            except ImportError:
                raise ImportError("Could not list repository files. Please update huggingface-hub.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download GGUF model {model_name}: {e}")
    
    
    def _init_hf_model(self, model_name: str):
        """Initialize HuggingFace model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generation config for HuggingFace models - use centralized settings
        self.generation_config = GenerationConfig(
            max_new_tokens=self.generation_settings["max_tokens"],
            temperature=self.generation_settings["temperature"],
            top_p=self.generation_settings["top_p"],
            top_k=self.generation_settings["top_k"],
            repetition_penalty=self.generation_settings["repetition_penalty"],
            do_sample=self.generation_settings["do_sample"],
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def format_conversation_history(self) -> str:
        """
        Format the conversation history according to training data structure.
        Returns the formatted conversation string.
        """
        if not self.conversation_history:
            return ""
        
        formatted_history = ""
        for entry in self.conversation_history:
            formatted_history += f"Student: {entry['student']}\n"
            formatted_history += f"Teacher: {entry['teacher']}\n"
        
        return formatted_history
    
    def truncate_history_if_needed(self, new_student_input: str) -> str:
        """
        Truncate conversation history if the total context would exceed max tokens.
        
        Args:
            new_student_input: The new student input to be added
            
        Returns:
            Formatted conversation history that fits within token limits
        """
        history_str = self.format_conversation_history()
        potential_instruction = f"{history_str}Student: {new_student_input}\nTeacher:"
        
        # Count tokens differently for GGUF vs HuggingFace
        if self.is_gguf:
            # For GGUF models, approximate token count
            token_count = len(potential_instruction.split()) * 1.3  # Rough approximation
        else:
            # For HuggingFace models, use tokenizer
            tokens = self.tokenizer.encode(potential_instruction, return_tensors="pt")
            token_count = tokens.shape[1]
        
        # If within limits, return as is
        if token_count <= self.max_context_tokens:
            return potential_instruction
        
        # Otherwise, truncate history from the beginning
        print(f"Context too long ({int(token_count)} tokens), truncating history")
        
        truncated_history = []
        for i in range(len(self.conversation_history) - 1, -1, -1):
            temp_history = [self.conversation_history[j] for j in range(i, len(self.conversation_history))]
            temp_str = ""
            for entry in temp_history:
                temp_str += f"Student: {entry['student']}\nTeacher: {entry['teacher']}\n"
            
            test_instruction = f"{temp_str}Student: {new_student_input}\nTeacher:"
            
            # Count tokens for test instruction
            if self.is_gguf:
                test_token_count = len(test_instruction.split()) * 1.3
            else:
                test_tokens = self.tokenizer.encode(test_instruction, return_tensors="pt")
                test_token_count = test_tokens.shape[1]
            
            if test_token_count <= self.max_context_tokens:
                truncated_history = temp_history
                break
        
        # Update the conversation history to truncated version
        self.conversation_history = truncated_history
        
        #Format truncated history
        truncated_str = ""
        for entry in truncated_history:
            truncated_str += f"Student: {entry['student']}\nTeacher: {entry['teacher']}\n"
        
        final_instruction = f"{truncated_str}Student: {new_student_input}\nTeacher:"
        
        # Final token count
        if self.is_gguf:
            final_token_count = len(final_instruction.split()) * 1.3
        else:
            final_tokens = self.tokenizer.encode(final_instruction, return_tensors="pt")
            final_token_count = final_tokens.shape[1]
        
        print(f"Context truncated to {int(final_token_count)} tokens")
        
        return final_instruction
    
    def generate_response(self, student_input: str) -> str:
        """
        Generate a teacher response for the given student input.
        
        Args:
            student_input: The student's question/message
            
        Returns:
            The teacher's response
        """
        # Prepare the instruction with conversation history
        instruction = self.truncate_history_if_needed(student_input)
        
        print(f"Generating response for: {student_input[:50]}{'...' if len(student_input) > 50 else ''}")
        
        if self.is_gguf:
            return self._generate_gguf_response(instruction)
        else:
            return self._generate_hf_response(instruction)
    
    def _generate_gguf_response(self, instruction: str) -> str:
        """Generate response using GGUF model."""
        try:
            # Generate with GGUF model
            response = self.model(
                instruction,
                **self.generation_params
            )
            
            generated_text = response['choices'][0]['text']
            
            # Extract teacher response
            full_text = instruction + generated_text
            return self._extract_teacher_response(full_text, instruction)
            
        except Exception as e:
            print(f"Error generating GGUF response: {e}")
            return "I'm not sure how to respond to that. [GGUF generation error]"
    
    def _generate_hf_response(self, instruction: str) -> str:
        """Generate response using HuggingFace model."""
        try:
            # Tokenize the instruction
            inputs = self.tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context_tokens
            )
            
            #Appropriate device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_teacher_response(generated_text, instruction)
            
        except Exception as e:
            print(f"Error generating HF response: {e}")
            return "I'm not sure how to respond to that. [HuggingFace generation error]"
    
    def _extract_teacher_response(self, generated_text: str, instruction: str) -> str:
        """Extract the teacher's response from generated text."""
        # Extract only the teacher's response
        if "Teacher:" in generated_text:
            # Find the last occurrence of "Teacher:" to get the generated response
            teacher_parts = generated_text.split("Teacher:")
            if len(teacher_parts) > 1:
                teacher_response = teacher_parts[-1].strip()
                # Clean up any potential continuation or unwanted text
                teacher_response = teacher_response.split("\n")[0].strip()
                if not teacher_response:
                    teacher_response = teacher_parts[-1].strip()
            else:
                teacher_response = "I'm not sure how to respond to that. [Fallback - no teacher response found]"
        else:
            # Fallback: extract text after the instruction
            instruction_end = generated_text.find(instruction) + len(instruction)
            teacher_response = generated_text[instruction_end:].strip()
            if not teacher_response:
                teacher_response = "I'm not sure how to respond to that. [Fallback - no response generated]"
        
        return teacher_response
    
    def chat_turn(self, student_input: str) -> str:
        """
        Process one complete chat turn: generate response and update history.
        
        Args:
            student_input: The student's input
            
        Returns:
            The teacher's response
        """
        teacher_response = self.generate_response(student_input)
        
        # Add to conversation history
        self.conversation_history.append({
            "student": student_input,
            "teacher": teacher_response
        })
        
        return teacher_response
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("Conversation history reset!")
    
    def save_conversation(self, filename: str):
        """Save the current conversation to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        print(f"Conversation saved to {filename}")
    
    def load_conversation(self, filename: str):
        """Load conversation history from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"Conversation loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found!")
        except json.JSONDecodeError:
            print(f"Error parsing {filename}!")
    
    def print_conversation_stats(self):
        """Print statistics about the current conversation."""
        if not self.conversation_history:
            print("No conversation history")
            return
            
        total_turns = len(self.conversation_history)
        history_str = self.format_conversation_history()
        
        # Count tokens differently for GGUF vs HuggingFace
        if self.is_gguf:
            token_count = int(len(history_str.split()) * 1.3)  # Approximate
        else:
            token_count = len(self.tokenizer.encode(history_str))
        
        print(f"Conversation Stats:")
        print(f"   • Total turns: {total_turns}")
        print(f"   • History tokens: {token_count} (approx)" if self.is_gguf else f"   • History tokens: {token_count}")
        print(f"   • Token limit: {self.max_context_tokens}")


def batch_questions_mode(chatbot: ConversationAwareChatbot, questions_file: str, output_filename: str = None):
    """
    Run batch questions mode with interactive UI display and incremental saving.
    
    Args:
        chatbot: The initialized chatbot instance
        questions_file: Path to JSON file containing questions
        output_filename: Optional output filename (if None, will prompt user)
    """
    try:
        # Load questions from file
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions from the data
        questions = []
        if isinstance(data, list):
            # Handle list format: [{"Question": "..."}, {"Question": "..."}, ...]
            for item in data:
                if isinstance(item, dict) and "Question" in item:
                    questions.append(item["Question"])
                elif isinstance(item, str):
                    questions.append(item)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    questions.append(value)
        
        if not questions:
            print("No questions found in the file. Please check the format.")
            print("Expected format: [{\"Question\": \"What are cats?\"}, {\"Question\": \"Who is Newton?\"}]")
            return
        
        total_questions = len(questions)
        print(f"\nLoaded {total_questions} questions from {questions_file}")
        print("="*60)
        print("BATCH PROCESSING MODE")
        print("="*60)
        
        # Handle output filename
        if output_filename:
            # Use provided filename
            if not output_filename.endswith('.json'):
                output_filename += '.json'
            
            # Test if we can write to this file
            try:
                with open(output_filename, 'w', encoding='utf-8') as test_file:
                    json.dump([], test_file)
                print(f"Will save results to: {output_filename}")
            except Exception as e:
                print(f"Error: Cannot write to '{output_filename}': {e}")
                return
        else:
            # Ask for output filename interactively
            print("Choose output filename for saving results:")
            while True:
                output_filename = input("Enter filename to save results (will auto-save after each question): ").strip()
                
                if output_filename:
                    # Add .json extension if not present
                    if not output_filename.endswith('.json'):
                        output_filename += '.json'
                    
                    # Test if we can write to this file
                    try:
                        with open(output_filename, 'w', encoding='utf-8') as test_file:
                            json.dump([], test_file)
                        print(f"Will save results to: {output_filename}")
                        break
                    except Exception as e:
                        print(f"Cannot write to '{output_filename}': {e}")
                        print("Please try a different filename.")
                else:
                    print("Please enter a valid filename.")
        
        print("\nProcessing questions with conversation context...")
        print("Results will be saved after each question to prevent data loss.")
        print("Press Ctrl+C at any time to stop gracefully\n")
                
        # Helper functions for metadata
        def get_model_name(chatbot):
            """Get model name for metadata - dynamically fetched from actual source."""
            if chatbot.is_gguf:
                # For GGUF models, return the actual model path that was used
                return getattr(chatbot, 'model_path', 'Unknown GGUF Model')
            else:
                # For HuggingFace models, try multiple sources for the model name
                model_name = getattr(chatbot.model.config, '_name_or_path', None)
                if not model_name:
                    model_name = getattr(chatbot.model.config, 'name_or_path', None)
                if not model_name:
                    model_name = getattr(chatbot.tokenizer, 'name_or_path', 'Unknown HF Model')
                return model_name
        
        def get_model_type(chatbot):
            """Get model type for metadata."""
            if chatbot.is_gguf:
                return "GGUF"
            else:
                return getattr(chatbot.model.config, 'model_type', 'Unknown')
        
        def get_generation_params(chatbot):
            """Get generation parameters for metadata - from centralized settings."""
            base_params = {
                "max_tokens": chatbot.generation_settings["max_tokens"],
                "temperature": chatbot.generation_settings["temperature"],
                "top_p": chatbot.generation_settings["top_p"],
                "top_k": chatbot.generation_settings["top_k"],
                "repetition_penalty": chatbot.generation_settings["repetition_penalty"],
                "do_sample": chatbot.generation_settings["do_sample"]  # Use the actual setting for both model types
            }
            
            if chatbot.is_gguf:
                # Add GGUF-specific parameters
                base_params.update({
                    "stop_sequences": chatbot.generation_params.get("stop", [])
                })
            else:
                # Add HuggingFace-specific parameters
                base_params.update({
                    "pad_token_id": getattr(chatbot.generation_config, 'pad_token_id', None),
                    "eos_token_id": getattr(chatbot.generation_config, 'eos_token_id', None)
                })
            
            return base_params
        
        metadata = {
            "metadata": {
                "model_name": get_model_name(chatbot),
                "model_type": get_model_type(chatbot),
                "is_gguf": chatbot.is_gguf,
                "generation_parameters": get_generation_params(chatbot),
                "context_settings": {
                    "max_context_tokens": chatbot.max_context_tokens,
                },
                "processing_info": {
                    "total_questions": total_questions,
                    "device": "CPU (GGUF)" if chatbot.is_gguf else ("CUDA" if torch.cuda.is_available() else "CPU")
                }
            },
            "results": []
        }
        
        # Save initial metadata
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Initialized results file with metadata: {output_filename}")
        except Exception as e:
            print(f"Error creating initial file: {e}")
            return
        
        # Process questions one by one with visual progress
        for i, question in enumerate(questions, 1):
            try:
                print(f"[{i}/{total_questions}] Processing question: {question[:60]}{'...' if len(question) > 60 else ''}")
                print(f"Student: {question}")
                
                # Generate response with progress indicator
                print("Thinking...", end="", flush=True)
                response = chatbot.chat_turn(question)
                print(f"Teacher: {response}")
                
                # Add this Q&A pair to results
                qa_pair = {
                    "question_number": i,
                    "student": question,
                    "teacher": response
                }
                metadata["results"].append(qa_pair)
                
                # Save incrementally after each question
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    print(f" Saved to {output_filename}")
                except Exception as save_error:
                    print(f"  Warning: Could not save to file: {save_error}")
                
                # Show progress bar
                progress = (i / total_questions) * 100
                bar_length = 30
                filled_length = int(bar_length * i // total_questions)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f" Progress: |{bar}| {progress:.1f}% ({i}/{total_questions})")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n  Processing interrupted after {i-1} questions!")
                print(f" Results saved to {output_filename} (contains {len(metadata['results'])} Q&A pairs)")
                break
            except Exception as e:
                print(f"\n Error processing question {i}: {e}")
                print("Continuing with next question...\n")
                continue
        
        # Final statistics
        processed_count = len(metadata["results"])
        print("="*60)
        print("🎉 BATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f" Processed: {processed_count} questions")
        print(f" All results saved to: {output_filename}")
        print(f" File contains {processed_count} question-answer pairs")
        
        # Show conversation stats
        chatbot.print_conversation_stats()
        
        # Create summary
        print(f"\n SUMMARY:")
        print(f"   • Input file: {questions_file}")
        print(f"   • Output file: {output_filename}")
        print(f"   • Questions processed: {processed_count}/{total_questions}")
        print(f"   • Conversation turns in memory: {len(chatbot.conversation_history)}")
        print(f"   • Model: {metadata['metadata']['model_name']}")
        print(f"   • Temperature: {metadata['metadata']['generation_parameters']['temperature']}")
        print(f"   • Context window: {metadata['metadata']['context_settings']['max_context_tokens']} tokens")
        
    except FileNotFoundError:
        print(f" Questions file '{questions_file}' not found!")
        print("Please check the file path and try again.")
    except json.JSONDecodeError as e:
        print(f" Error parsing JSON file: {e}")
        print("Please check the file format. Expected: [{\"Question\": \"...\"}, {\"Question\": \"...\"}]")
    except Exception as e:
        print(f" Unexpected error: {e}")


def interactive_chat(chatbot: ConversationAwareChatbot):
    """Run interactive chat session."""
    print("\n" + "="*60)
    print("CONVERSATION-AWARE TEACHER CHATBOT")
    print("="*60)
    print("Commands:")
    print("  • /reset - Reset conversation history")
    print("  • /save <filename> - Save conversation to file")
    print("  • /load <filename> - Load conversation from file") 
    print("  • /stats - Show conversation statistics")
    print("  • /quit or /exit - Exit the chat")
    print("  • Just type your question to chat!")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("🎓 Student: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                command_parts = user_input.split(' ', 1)
                command = command_parts[0].lower()
                
                if command in ['/quit', '/exit']:
                    print("Goodbye! Thanks for chatting!")
                    break
                elif command == '/reset':
                    chatbot.reset_conversation()
                elif command == '/stats':
                    chatbot.print_conversation_stats()
                elif command == '/save':
                    filename = command_parts[1] if len(command_parts) > 1 else 'conversation.json'
                    chatbot.save_conversation(filename)
                elif command == '/load':
                    filename = command_parts[1] if len(command_parts) > 1 else 'conversation.json'
                    chatbot.load_conversation(filename)
                else:
                    print("Unknown command. Type /quit to exit.")
                continue
            
            # Generate and display response
            print(" Thinking...", end="", flush=True)
            response = chatbot.chat_turn(user_input)
            print(f"\rTeacher: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type /quit to exit.\n")


def main():
    parser = argparse.ArgumentParser(description="Conversation-Aware Teacher Chatbot")
    parser.add_argument(
        "--model", 
        type=str, 
        default="MebinThattil/FT-LlaMA-135-Claude_Sonnet_4-Distill-RUN1-Quantized4Bit-GGUF",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=512,
        help="Maximum context tokens (default: 512)"
    )
    parser.add_argument(
        "--single", 
        type=str, 
        help="Single question mode - provide question as argument"
    )
    parser.add_argument(
        "--load-conversation", 
        type=str, 
        help="Load conversation history from JSON file"
    )
    parser.add_argument(
        "--batch-questions", 
        type=str, 
        help="Batch questions mode - provide JSON file with list of questions"
    )
    parser.add_argument(
        "--save", 
        type=str, 
        help="Output filename for batch mode results (auto-adds .json extension if not present)"
    )
    parser.add_argument(
        "--mode", 
        type=int, 
        choices=[1, 2, 3],
        default=1,
        help="Generation mode: 1=default (temp=0.7, do_sample=True), 2=low_temp (temp=0.3, do_sample=True), 3=deterministic (temp=0.7, do_sample=False)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = ConversationAwareChatbot(args.model, args.max_tokens, args.mode)
        
        # Load conversation if specified
        if args.load_conversation:
            chatbot.load_conversation(args.load_conversation)
        
        # Single question mode
        if args.single:
            response = chatbot.chat_turn(args.single)
            print(f"Student: {args.single}")
            print(f"Teacher: {response}")
            return
        
        # Batch questions mode
        if args.batch_questions:
            batch_questions_mode(chatbot, args.batch_questions, args.save)
            return
        
        # Interactive mode
        interactive_chat(chatbot)
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()