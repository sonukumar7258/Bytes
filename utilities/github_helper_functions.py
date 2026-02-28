import os
import json

def flatten_repo_data(all_repos_data):
    """
    Flatten repository data into a list of strings, each representing an issue or commit.

    Args:
        all_repos_data (list): List of dictionaries containing data for each repository.

    Returns:
        list: List of strings, each representing an issue or commit.

    """
    chunks = []
    for repo_data in all_repos_data:
        repo_name = f"repo_name:{repo_data['name']}"
        repo_description = f"repo_description:{repo_data['description']}"

        # Flatten issues data
        for issue in repo_data['issues']:
            issue_line = " ".join([
                "github\n",
                "Issue:",
                repo_name,
                repo_description,
                f"issue_title:{issue['title']}",
                f"issue_number:{issue['number']}",
                f"issue_state:{issue['state']}",
                f"created_at:{issue['created_at']}",
                f"updated_at:{issue['updated_at']}",
                f"url:{issue['url']}",
            ])
            chunks.append(issue_line)

        # Flatten commits data
        for branch in repo_data['branches']:
            branch_name = f"branch_name:{branch['name']}"
            for commit in branch['commits']:
                commit_message_cleaned = commit['message'].replace('\n', ' ')
                commit_line = " ".join([
                    "github\n",
                    "Commit:",
                    repo_name,
                    repo_description,
                    branch_name,
                    f"commit_message:{commit_message_cleaned}",
                    f"commit_date:{commit['date']}",
                    f"commit_author:{commit['author']}",
                    f"url:{commit['url']}"
                ])
                chunks.append(commit_line)

    return chunks
