#!/usr/bin/env python3
"""
RAG Chatbot Interface
A terminal-based chatbot that connects to your FastAPI RAG system for interactive conversations.

Usage:
    python chatbot.py
    python chatbot.py --host localhost --port 8000
    python chatbot.py --save-history
"""

import sys
import argparse
import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import readline  # For better input editing

# ANSI color codes for better terminal experience
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.RESET}")

def print_banner():
    """Display welcome banner."""
    print_colored("=" * 80, Colors.CYAN)
    print_colored("🤖 RAG CHATBOT - Interactive LangChain Assistant", Colors.BOLD + Colors.CYAN)
    print_colored("=" * 80, Colors.CYAN)
    print_colored("💬 Start chatting with your RAG system!", Colors.GREEN)
    print_colored("💡 Type 'help' for commands, 'quit' or 'exit' to leave", Colors.YELLOW)
    print_colored("-" * 80, Colors.CYAN)

def check_api_health(api_base_url: str) -> bool:
    """Check if the FastAPI server is running and healthy."""
    try:
        print_colored("🔍 Checking API health...", Colors.YELLOW)
        response = requests.get(f"{api_base_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_colored("✅ API is healthy!", Colors.GREEN)
            print_colored(f"   Service: {data.get('service', 'Unknown')}", Colors.WHITE)
            print_colored(f"   Documents: {data.get('documents_count', 0):,}", Colors.WHITE)
            print_colored(f"   Database: {'✅' if data.get('database_connected') else '❌'}", Colors.WHITE)
            print_colored(f"   LLM Model: {data.get('llm_model', 'Unknown')}", Colors.WHITE)
            return True
        else:
            print_colored(f"❌ API health check failed (Status: {response.status_code})", Colors.RED)
            return False
            
    except requests.ConnectionError:
        print_colored("❌ Cannot connect to API server", Colors.RED)
        print_colored("💡 Make sure FastAPI server is running:", Colors.YELLOW)
        print_colored("   source .venv/bin/activate && python app/main.py", Colors.WHITE)
        return False
    except Exception as e:
        print_colored(f"❌ API health check error: {str(e)}", Colors.RED)
        return False

class RAGChatbot:
    """Main chatbot class for interacting with RAG API."""
    
    def __init__(self, api_base_url: str, save_history: bool = False):
        self.api_base_url = api_base_url
        self.save_history = save_history
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        self.session_stats = {
            "start_time": datetime.now(),
            "total_queries": 0,
            "successful_responses": 0,
            "total_response_time": 0.0,
            "topics_discussed": set()
        }
        
        # Initialize history file if saving
        if self.save_history:
            self.history_file = Path(f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            self._init_history_file()
    
    def _init_history_file(self):
        """Initialize the chat history markdown file."""
        header = f"""# RAG Chatbot Session History

**Session ID:** {self.conversation_id}  
**Start Time:** {self.session_stats['start_time'].isoformat()}  
**API Endpoint:** {self.api_base_url}  

---

"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            f.write(header)
        
        print_colored(f"📝 Saving chat history to: {self.history_file}", Colors.CYAN)
    
    def _save_to_history(self, user_query: str, bot_response: Dict[str, Any], response_time: float):
        """Save conversation exchange to history file."""
        if not self.save_history:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        exchange = f"""## 💬 Exchange #{len(self.conversation_history)}

**Time:** {timestamp}  
**Response Time:** {response_time:.1f}ms  

### 👤 You:
{user_query}

### 🤖 Assistant:
{bot_response.get('answer', 'No response')}

"""
        
        # Add sources if available
        citations = bot_response.get('citations', [])
        if citations:
            exchange += f"""### 📚 Sources ({len(citations)}):
"""
            for i, citation in enumerate(citations, 1):
                file_name = citation.get('file_name', 'Unknown')
                doc_type = citation.get('doc_type', 'Unknown')
                score = citation.get('similarity_score', 0)
                exchange += f"- **[{i}]** {file_name} ({doc_type}) - Score: {score:.4f}\n"
        
        # Add follow-up questions
        followup = bot_response.get('follow_up_questions', [])
        if followup:
            exchange += f"""
### 🤔 Follow-up Questions:
"""
            for i, question in enumerate(followup, 1):
                exchange += f"{i}. {question}\n"
        
        exchange += "\n---\n\n"
        
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(exchange)
        except Exception as e:
            print_colored(f"⚠️  Could not save to history: {e}", Colors.YELLOW)
    
    def _update_stats(self, query: str, success: bool, response_time: float):
        """Update session statistics."""
        self.session_stats["total_queries"] += 1
        if success:
            self.session_stats["successful_responses"] += 1
        self.session_stats["total_response_time"] += response_time
        
        # Extract topics (simple keyword extraction)
        keywords = [word.lower() for word in query.split() 
                   if len(word) > 4 and word.isalpha()]
        self.session_stats["topics_discussed"].update(keywords[:3])  # Keep top 3 keywords
    
    def query_rag(self, question: str) -> Dict[str, Any]:
        """Send query to RAG API and get response."""
        try:
            payload = {
                "question": question,
                "conversation_id": self.conversation_id,
                "conversation_mode": "multi_turn",
                "top_k": 5,
                "use_citations": True,
                "generate_followup": True,
                "max_response_tokens": 4000
            }
            
            print_colored("🔍 Thinking...", Colors.YELLOW)
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/api/v1/rag/query",
                json=payload,
                timeout=60,  # 60 second timeout
                headers={"Content-Type": "application/json"}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self._update_stats(question, True, response_time)
                return {
                    **data,
                    "response_time_ms": response_time
                }
            else:
                error_msg = f"API request failed (Status: {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', error_msg)
                except:
                    pass
                
                self._update_stats(question, False, response_time)
                return {
                    "success": False,
                    "error": error_msg,
                    "response_time_ms": response_time
                }
                
        except requests.Timeout:
            print_colored("⏰ Request timed out (60s)", Colors.RED)
            return {"success": False, "error": "Request timeout"}
        except requests.ConnectionError:
            print_colored("❌ Connection error - Is the API server running?", Colors.RED)
            return {"success": False, "error": "Connection error"}
        except Exception as e:
            print_colored(f"❌ Unexpected error: {str(e)}", Colors.RED)
            return {"success": False, "error": str(e)}
    
    def display_response(self, response: Dict[str, Any]):
        """Display the RAG response in a formatted way."""
        if not response.get("success", False):
            print_colored(f"❌ Error: {response.get('error', 'Unknown error')}", Colors.RED)
            return
        
        # Main answer
        answer = response.get("answer", "No answer provided")
        print_colored("\n🤖 Assistant:", Colors.BOLD + Colors.BLUE)
        print_colored(answer, Colors.WHITE)
        
        # Citations
        citations = response.get("citations", [])
        if citations:
            print_colored(f"\n📚 Sources ({len(citations)}):", Colors.BOLD + Colors.CYAN)
            for i, citation in enumerate(citations, 1):
                file_name = citation.get("file_name", "Unknown")
                doc_type = citation.get("doc_type", "Unknown")
                score = citation.get("similarity_score", 0)
                print_colored(f"  [{i}] {file_name} ({doc_type}) - {score:.4f}", Colors.CYAN)
        
        # Follow-up questions
        followup = response.get("follow_up_questions", [])
        if followup:
            print_colored(f"\n💡 You might also ask:", Colors.BOLD + Colors.YELLOW)
            for i, question in enumerate(followup, 1):
                print_colored(f"  {i}. {question}", Colors.YELLOW)
        
        # Performance info
        response_time = response.get("response_time_ms", 0)
        confidence = response.get("confidence_score", 0)
        if response_time > 0:
            print_colored(f"\n⏱️  Response time: {response_time:.0f}ms", Colors.MAGENTA)
            if confidence > 0:
                print_colored(f"🎯 Confidence: {confidence:.2f}", Colors.MAGENTA)
    
    def show_help(self):
        """Display help information."""
        print_colored("\n📖 CHATBOT COMMANDS:", Colors.BOLD + Colors.CYAN)
        print_colored("  help, h        - Show this help message", Colors.WHITE)
        print_colored("  stats          - Show session statistics", Colors.WHITE)
        print_colored("  history        - Show conversation history", Colors.WHITE)
        print_colored("  clear          - Clear screen", Colors.WHITE)
        print_colored("  save           - Save current history to file", Colors.WHITE)
        print_colored("  new, reset     - Start new conversation", Colors.WHITE)
        print_colored("  quit, exit, q  - Exit chatbot", Colors.WHITE)
        print_colored("\n💬 Just type your question to chat with the RAG system!", Colors.GREEN)
    
    def show_stats(self):
        """Display session statistics."""
        duration = datetime.now() - self.session_stats["start_time"]
        avg_response_time = (self.session_stats["total_response_time"] / 
                           max(self.session_stats["total_queries"], 1))
        success_rate = (self.session_stats["successful_responses"] / 
                       max(self.session_stats["total_queries"], 1)) * 100
        
        print_colored("\n📊 SESSION STATISTICS:", Colors.BOLD + Colors.CYAN)
        print_colored(f"  Session Duration: {duration}", Colors.WHITE)
        print_colored(f"  Total Queries: {self.session_stats['total_queries']}", Colors.WHITE)
        print_colored(f"  Successful Responses: {self.session_stats['successful_responses']}", Colors.WHITE)
        print_colored(f"  Success Rate: {success_rate:.1f}%", Colors.WHITE)
        print_colored(f"  Average Response Time: {avg_response_time:.0f}ms", Colors.WHITE)
        print_colored(f"  Conversation ID: {self.conversation_id}", Colors.WHITE)
        
        if self.session_stats["topics_discussed"]:
            topics = ", ".join(list(self.session_stats["topics_discussed"])[:5])
            print_colored(f"  Topics Discussed: {topics}", Colors.WHITE)
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print_colored("📝 No conversation history yet", Colors.YELLOW)
            return
        
        print_colored(f"\n📜 CONVERSATION HISTORY ({len(self.conversation_history)} exchanges):", 
                     Colors.BOLD + Colors.CYAN)
        
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            timestamp = exchange.get("timestamp", "")
            user_msg = exchange.get("user_message", "")[:100]
            bot_msg = exchange.get("bot_response", {}).get("answer", "")[:100]
            
            print_colored(f"\n  {i}. [{timestamp}]", Colors.MAGENTA)
            print_colored(f"     You: {user_msg}{'...' if len(user_msg) >= 100 else ''}", Colors.WHITE)
            print_colored(f"     Bot: {bot_msg}{'...' if len(bot_msg) >= 100 else ''}", Colors.WHITE)
        
        if len(self.conversation_history) > 5:
            print_colored(f"\n  ... and {len(self.conversation_history) - 5} more exchanges", Colors.YELLOW)
    
    def new_conversation(self):
        """Start a new conversation."""
        self.conversation_id = str(uuid.uuid4())
        old_count = len(self.conversation_history)
        self.conversation_history = []
        print_colored(f"🔄 Started new conversation (saved {old_count} previous exchanges)", Colors.GREEN)
        print_colored(f"🆔 New conversation ID: {self.conversation_id[:8]}...", Colors.CYAN)
    
    def run(self):
        """Main chatbot loop."""
        print_banner()
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input(f"\n{Colors.BOLD}{Colors.GREEN}💬 You: {Colors.RESET}").strip()
                except KeyboardInterrupt:
                    print_colored("\n\n👋 Goodbye!", Colors.CYAN)
                    break
                except EOFError:
                    print_colored("\n\n👋 Goodbye!", Colors.CYAN)
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                cmd = user_input.lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    print_colored("👋 Goodbye!", Colors.CYAN)
                    break
                
                elif cmd in ['help', 'h']:
                    self.show_help()
                    continue
                
                elif cmd == 'stats':
                    self.show_stats()
                    continue
                
                elif cmd == 'history':
                    self.show_history()
                    continue
                
                elif cmd == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_banner()
                    continue
                
                elif cmd in ['new', 'reset']:
                    self.new_conversation()
                    continue
                
                elif cmd == 'save':
                    if self.save_history:
                        print_colored(f"💾 History automatically saved to: {self.history_file}", Colors.GREEN)
                    else:
                        print_colored("💡 Use --save-history flag when starting to enable auto-save", Colors.YELLOW)
                    continue
                
                # Process RAG query
                response = self.query_rag(user_input)
                self.display_response(response)
                
                # Save to conversation history
                exchange = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_message": user_input,
                    "bot_response": response
                }
                self.conversation_history.append(exchange)
                
                # Save to file if enabled
                if response.get("success", False):
                    response_time = response.get("response_time_ms", 0)
                    self._save_to_history(user_input, response, response_time)
        
        except Exception as e:
            print_colored(f"\n❌ Chatbot error: {str(e)}", Colors.RED)
        
        finally:
            # Final stats
            print_colored("\n" + "="*50, Colors.CYAN)
            self.show_stats()
            if self.save_history and hasattr(self, 'history_file'):
                print_colored(f"💾 Full history saved to: {self.history_file}", Colors.GREEN)
            print_colored("="*50, Colors.CYAN)

def main():
    """Main function for the chatbot."""
    parser = argparse.ArgumentParser(
        description="Interactive RAG Chatbot - Chat with your LangChain system via FastAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py
  python chatbot.py --host localhost --port 8000
  python chatbot.py --save-history
  python chatbot.py --host 127.0.0.1 --port 8080 --save-history
        """
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="FastAPI server host (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000)"
    )
    
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save chat history to markdown file"
    )
    
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip API health check on startup"
    )
    
    args = parser.parse_args()
    
    # Build API URL
    api_base_url = f"http://{args.host}:{args.port}"
    
    # Health check
    if not args.no_health_check:
        if not check_api_health(api_base_url):
            print_colored("\n❌ Cannot proceed without healthy API connection", Colors.RED)
            print_colored("💡 Start the FastAPI server first:", Colors.YELLOW)
            print_colored("   source .venv/bin/activate && python app/main.py", Colors.WHITE)
            print_colored("   Or use --no-health-check to skip this check", Colors.WHITE)
            sys.exit(1)
    
    # Start chatbot
    try:
        chatbot = RAGChatbot(api_base_url, args.save_history)
        chatbot.run()
    except KeyboardInterrupt:
        print_colored("\n\n👋 Chatbot interrupted by user", Colors.CYAN)
    except Exception as e:
        print_colored(f"\n❌ Failed to start chatbot: {str(e)}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()