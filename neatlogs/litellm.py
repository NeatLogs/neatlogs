from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from .log_parser import add_parser_to_provider
import json
import requests
from dataclasses import asdict
import threading


# Define a dataclass to structure the logging of LLM interactions
# Dataclasses automatically generate __init__, __repr__, etc.
@dataclass
class LLMEvent:
    """
    Records a single interaction between the agent and the LLM.
    This class structures the data before being processed by the log parser.
    """
    event_type: str = "llms"
    prompt: Optional[Union[str, List]] = None
    completion: Optional[Union[str, Dict]] = None
    model: Optional[str] = None
    timestamp: str = None
    metadata: Dict = None

class LiteLLMProvider:
    """
    Handles the interception and logging of LLM interactions.
    Simplified to focus on conversation logging functionality.
    """
    def __init__(self, trace_id, client=None, api_key=None, tags=None):
        """
        Initialize the provider with a log file.
        
        Args:
            trace_id (str): Unique identifier for the tracking session
            client (str, optional): Client identifier
            api_key (str, optional): API key for authentication
            tags (dict, optional): Dictionary of tags to associate with the tracking session
        """
        self.original_create = None #original method that we are patching
        self.conversation_ring = []
        self.max_history = 100  # Store last 100 conversations in memory
        self.client = client
        self.trace_id = trace_id
        self.api_key = api_key
        self.tags = tags or []

        '''        
        Ring Buffer:

            It's a fixed-size list that overwrites old data when full
            Like a circular queue - when it reaches max size (100 in this case), new items replace the oldest ones
            Example: With max_size=3:
            pythonCopy# Start: []
            # Add A: [A]
            # Add B: [A,B] 
            # Add C: [A,B,C]
            # Add D: [B,C,D] (A gets removed)


            Storing in Memory:

            Data is kept in RAM (temporary) rather than written to disk (permanent)
            Exists only while program runs, cleared when program ends
            Faster access but temporary storage
            In this code, recent conversations are kept in memory for quick retrieval without reading from files

            The key difference is:

            Files (log_parser.py) = permanent storage on disk
            Ring buffer (conversation_ring) = temporary storage in RAM for quick access to recent items

            When someone calls get_conversation_history(), they can instantly get recent conversations without reading from files
            Useful for displaying recent history in UI or for rapid analysis of latest interactions
        '''

    def override(self):
        """
        Override LiteLLM's completion method to enable logging.
        This method patches the original completion function with our logging version.

        Inside __init__ of LiteLLMProvider, there are 3 key elements:

        self.original_create = None - Stores the original litellm completion method for later restoration
        self.conversation_ring = [] - A list acting as a ring buffer to store recent conversations
        self.max_history = 100 - Maximum number of conversations to keep in memory
        """
        import litellm
        
        # Store the original method so we can restore it later
        self.original_create = litellm.completion
        
        # Define our patched version of the completion method
        def patched_sync(*args, **kwargs):
            """
            Wrapped version of completion that includes logging.
            """
            print("\n===== Input =====")
            print("\nArgs (Positional Arguments):")
            if args:
                for i, arg in enumerate(args):
                    print(f"Arg {i}: {arg}")
            else:
                print("No positional arguments")

            print("\nKwargs (Keyword Arguments):")
            if kwargs:
                for key, value in kwargs.items():
                    print(f"{key}: {value}")
            else:
                print("No keyword arguments")
                
            # Call original method to get response
            try:
                response = self.original_create(*args, **kwargs)
            except Exception as e:
                self.handle_error(e, kwargs)
                raise e
            
            print("\n===== Output =====")
            print(f"Response: {response}")
            print("=" * 50 + "\n")
            
            # Log the interaction and return the response
            return self._log_interaction(kwargs, response)
        
        # Replace original method with our patched version
        litellm.completion = patched_sync

    def _log_interaction(self, kwargs: dict, response: Any):
        """
        Pre-processing hook that structures data before the log parser processes it.
        """
        # Extract basic information
        save_data_in_neat_logs(kwargs, response, self.trace_id, self.api_key, tags=self.tags)
        prompt = kwargs.get("messages", kwargs.get("prompt", "N/A"))
        model = response.model if hasattr(response, "model") else "unknown"
        completion = response.choices[0].message.content if hasattr(response, "choices") else str(response)

        # Create event object
        event = LLMEvent(
            prompt=prompt,
            completion=completion,
            model=model,
            timestamp=datetime.now().isoformat(),
            metadata={
                "total_tokens": response.usage.total_tokens if hasattr(response, "usage") else 0,
                "tags": self.tags
            }
        )
        
        # Update conversation ring buffer
        if len(self.conversation_ring) >= self.max_history:
            self.conversation_ring.pop(0)
        self.conversation_ring.append(event)
        
        return response

    def undo_override(self):
        """
        Restore the original LiteLLM completion method.
        Should be called when logging is no longer needed.
        """
        import litellm
        
        if self.original_create:
            litellm.completion = self.original_create

    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """
        Returns recent conversation history with optional limit.
        """
        history = self.conversation_ring[-limit:] if limit else self.conversation_ring
        return [
            {
                "timestamp": event.timestamp,
                "model": event.model,
                "prompt": event.prompt,
                "completion": event.completion,
                "metadata": event.metadata
            }
            for event in history
        ]
    
    def handle_error(self, e, kwargs):
        """
        Handle errors in the original method.
        """
        print(f"Error in original method: {e}")
        save_data_in_neat_logs(kwargs, None, self.trace_id, self.api_key, error=e, tags=self.tags)
        return None


# Apply the parser decorator to the provider class
add_parser_to_provider(LiteLLMProvider)


def _save_data_in_background(kwargs: dict, response: Any, trace_id, api_key, error=None, tags=None):
    """Background thread function to send data to server."""
    url = "https://app.neatlogs.com/api/data"
    headers = {"Content-Type": "application/json"}
    try:
        if hasattr(response, "dict"):
            json_data = response.dict()  # Pydantic v1
        elif hasattr(response, "model_dump"):
            json_data = response.model_dump()  # Pydantic v2
        elif hasattr(response, "__dict__"):
            json_data = vars(response)  # Regular class
        elif isinstance(response, tuple):  # NamedTuple
            json_data = response._asdict()
        elif hasattr(response, "__dataclass_fields__"):  # Dataclass
            json_data = asdict(response)
        else:
            raise TypeError("Cannot serialize object")

        trace_data = {
            "kwargs": json.dumps(kwargs),
            "response": json.dumps(json_data)
        }

        # Convert error to a serializable format
        error_info = None
        if error is not None:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "args": getattr(error, 'args', None)
            }

        api_data = {
            "dataDump": json.dumps(trace_data),
            "projectAPIKey": api_key,
            "externalTraceId": trace_id,
            "error": error_info,  # Use the serialized error info instead
            "tags": tags or [],  # Ensure tags is a list
            "timestamp": datetime.now().timestamp()
        }

        requests.post(url, json=api_data, headers=headers)

    except Exception as e:
        print("Error in sending logs:", e)

def save_data_in_neat_logs(kwargs: dict, response: Any, trace_id, api_key, error=None, tags=None):
    """
    Non-blocking function that sends data to the server in a background thread.
    The thread will continue running even after the main program exits.
    
    Args:
        kwargs (dict): The input arguments for the LLM call
        response (Any): The response from the LLM
        trace_id (str): Unique identifier for the tracking session
        api_key (str): API key for authentication
        error (Exception, optional): Any error that occurred
        tags (dict, optional): Dictionary of tags to associate with the tracking session
    """
    thread = threading.Thread(
        target=_save_data_in_background,
        args=(kwargs, response, trace_id, api_key, error, tags),
        daemon=False  # Set to False so the thread continues after main program exits
    )
    thread.start()

# Example usage
# if __name__ == "__main__":
#     # Create a provider instance
#     provider = LiteLLMProvider()
#
#     # Enable logging
#     provider.override()
#
#     #others
#
#     # print("HERE HERE HERE HERE \n")
#     # print(provider.conversation_ring)  # List of recent LLM interactions
#     # history = provider.get_conversation_history(limit = 10)
#     # print(history)  # List of dictionaries with recent LLM interactions
#
#     # At this point, any litellm.completion calls will be logged
#     # Example:
#     # response = litellm.completion(prompt="Hello!")
#
#     # To get conversation history:
#     # history = provider.get_conversation_history()
#
#     # To disable logging:
#     # provider.undo_override()




