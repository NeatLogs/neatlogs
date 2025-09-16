"""
CrewAI Event Handler
====================

Handles CrewAI specific API patterns for comprehensive workflow and agent-level tracing.
"""

import json
import logging
from functools import wraps
from ..core import LLMSpan, current_span_id_context, set_current_agent_name, clear_current_agent_name
from .base import BaseEventHandler


class CrewAIHandler(BaseEventHandler):
    """
    Event handler for CrewAI.

    This handler wraps the `kickoff` method for overall workflow tracing and
    the `execute_task` method for individual agent-level tracing, creating a
    complete and connected graph of the crew's execution.
    """

    def __init__(self, tracker):
        super().__init__(tracker)

    def wrap_kickoff(self, original_kickoff):
        """Wraps `Crew.kickoff` to create a root workflow span."""
        @wraps(original_kickoff)
        def tracked_kickoff(crew_self, *args, **kwargs):
            if not self.tracker:
                return original_kickoff(crew_self, *args, **kwargs)

            workflow_span = self.tracker.start_llm_span(
                model="crew_workflow",
                provider="crewai_framework",
                framework="crewai",
                node_type="workflow",
                node_name="Crew Kickoff"
            )
            token = current_span_id_context.set(workflow_span.span_id)

            inputs = kwargs.get('inputs', {})
            try:
                workflow_span.messages = [{"role": "user", "content": json.dumps(inputs)}]
            except:
                workflow_span.messages = [{"role": "user", "content": str(inputs)}]

            try:
                result = original_kickoff(crew_self, *args, **kwargs)
                try:
                    workflow_span.completion = json.dumps(result)
                except:
                    workflow_span.completion = str(result)
                
                if workflow_span.end_time is None:
                    self.tracker.end_llm_span(workflow_span, success=True)
                
                return result
            except Exception as e:
                if workflow_span.end_time is None:
                    self.tracker.end_llm_span(workflow_span, success=False, error=e)
                raise
            finally:
                current_span_id_context.reset(token)
        
        return tracked_kickoff

    def wrap_agent_execute(self, original_execute):
        """Wraps `Agent.execute_task` to create a nested agent-level span."""
        @wraps(original_execute)
        def tracked_execute(agent_self, *args, **kwargs):
            if not self.tracker:
                return original_execute(agent_self, *args, **kwargs)

            agent_span = self.tracker.start_llm_span(
                model=f"agent/{agent_self.role}",
                provider="crewai_framework",
                framework="crewai",
                node_type="agent_execution",
                node_name=agent_self.role
            )
            
            span_token = current_span_id_context.set(agent_span.span_id)
            set_current_agent_name(agent_self.role)

            try:
                task = next((arg for arg in args if hasattr(arg, 'description')), kwargs.get('task'))
                if task:
                    agent_span.messages = [{"role": "user", "content": task.description}]

                result = original_execute(agent_self, *args, **kwargs)
                
                if isinstance(result, str):
                    agent_span.completion = result
                else:
                    agent_span.completion = str(result)

                if agent_span.end_time is None:
                    self.tracker.end_llm_span(agent_span, success=True)
                return result
            except Exception as e:
                if agent_span.end_time is None:
                    self.tracker.end_llm_span(agent_span, success=False, error=e)
                raise
            finally:
                clear_current_agent_name()
                current_span_id_context.reset(span_token)

        return tracked_execute
