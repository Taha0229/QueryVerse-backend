import pandas as pd
from datetime import datetime
import os
from query_verse.config import BASE_DIR

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate


class ConversationHistoryManager:
    def __init__(
        self, file_path=f"{BASE_DIR}/src/query_verse/db/conversation_history.csv"
    ):
        self.file_path = file_path
        self.columns = ["thread_id", "chat_title", "time"]
        self.initialize_file()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def initialize_file(self):
        """Create the CSV file with required columns if it doesn't exist."""
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.file_path, index=False)

    def add_conversation(self, thread_id, chat):
        """
        Add a new conversation to the history.

        Args:
            thread_id (str): Unique identifier for the conversation
            chat_title (str, optional): Name of the chat. If None, uses "New Chat"

        Returns:
            bool: True if successful, False if thread_id already exists
        """
        try:
            # Read existing data
            df = pd.read_csv(self.file_path)

            # Check if thread_id already exists
            if thread_id in df["thread_id"].values:
                return {}

            # Create new entry
            chat_title = self.llm.invoke(
                f"Generate a short chat title for the following message, it must not exceed 5 words: {chat}"
            ).content
            
            new_entry = {
                "thread_id": thread_id,
                "chat_title": chat_title,
                "time": datetime.now().isoformat(),
            }

            # Append new entry
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

            # Save updated dataframe
            df.to_csv(self.file_path, index=False)
            return new_entry

        except Exception as e:
            print(f"Error adding conversation: {e}")
            return {}

    def update_chat_name(self, thread_id, new_name):
        """
        Update the name of an existing chat.

        Args:
            thread_id (str): Thread ID of the chat to update
            new_name (str): New name for the chat

        Returns:
            bool: True if successful, False if thread_id not found
        """
        try:
            df = pd.read_csv(self.file_path)

            # Check if thread_id exists
            if thread_id not in df["thread_id"].values:
                return False

            # Update name
            df.loc[df["thread_id"] == thread_id, "chat_title"] = new_name

            # Save updated dataframe
            df.to_csv(self.file_path, index=False)
            return True

        except Exception as e:
            print(f"Error updating chat name: {e}")
            return False

    def get_conversation_history(self, limit=None, sort_by="time", ascending=False):
        """
        Retrieve conversation history.

        Args:
            limit (int, optional): Number of records to return
            sort_by (str, optional): Column to sort by
            ascending (bool, optional): Sort order

        Returns:
            pandas.DataFrame: Conversation history
        """
        try:
            df = pd.read_csv(self.file_path)

            # Sort the dataframe
            df = df.sort_values(by=sort_by, ascending=ascending)

            # Apply limit if specified
            if limit:
                df = df.head(limit)

            return df

        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return pd.DataFrame(columns=self.columns)

    def delete_conversation(self, thread_id):
        """
        Delete a conversation from the history.

        Args:
            thread_id (str): Thread ID of the chat to delete

        Returns:
            bool: True if successful, False if thread_id not found
        """
        try:
            df = pd.read_csv(self.file_path)

            # Check if thread_id exists
            if thread_id not in df["thread_id"].values:
                return False

            # Remove the conversation
            df = df[df["thread_id"] != thread_id]

            # Save updated dataframe
            df.to_csv(self.file_path, index=False)
            return True

        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize the manager

    manager = ConversationHistoryManager()

    # # Add some sample conversations
    # manager.add_conversation("123", "Python Discussion")
    # manager.add_conversation("456", "JavaScript Help")

    # Update a chat name
    manager.update_chat_name("123", "Advanced Python Discussion")

    # Get recent conversations
    recent_chats = manager.get_conversation_history(limit=5)
    print("\nRecent conversations:")
    print(recent_chats)

    # Delete a conversation
    manager.delete_conversation("456")

    # Show updated history
    print("\nAfter deletion:")
    print(manager.get_conversation_history())
