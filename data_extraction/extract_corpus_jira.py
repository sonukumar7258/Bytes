from jira import JIRA
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Load JIRA credentials from environment variables
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_INSTANCE_URL = os.getenv("JIRA_INSTANCE_URL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

print(JIRA_USERNAME)

# Initialize JIRA client
jira = JIRA(server=JIRA_INSTANCE_URL, basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN))

def create_corpus():
    """
    Create a corpus containing data from JIRA projects and issues.

    Returns:
        list: List of dictionaries containing data for each project and its issues.

    """
    corpus = []
    projects = jira.projects()

    # Iterate through each project
    for project in projects:
        project_data = {
            "project_key": project.key,
            "project_name": project.name,
            "issues": []
        }

        # Retrieve issues for the project
        issues = jira.search_issues(f'project={project.key}', maxResults=100)

        # Iterate through each issue
        for issue in issues:
            issue_data = {
                "issue_key": issue.key,
                "issue_summary": issue.fields.summary,
                "issue_type": issue.fields.issuetype.name,
                "issue_status": issue.fields.status.name,
                "comments": []
            }

            # Retrieve comments for the issue
            comments = jira.comments(issue)

            # Iterate through each comment
            for comment in comments:
                comment_data = {
                    "comment_id": comment.id,
                    "comment_author": comment.author.displayName,
                    "comment_body": comment.body
                }
                issue_data["comments"].append(comment_data)

            project_data["issues"].append(issue_data)

        corpus.append(project_data)

    return corpus

if __name__ == "__main__":
    # Create JIRA corpus
    corpus = create_corpus()

    # Save JIRA corpus to a JSON file
    corpus_directory = os.path.join(os.path.dirname(__file__), '../corpus/')
    corpus_file_path = os.path.join(corpus_directory, 'jira_corpus.json')

    with open(corpus_file_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
    
    print("Corpus has been extracted and saved.")
