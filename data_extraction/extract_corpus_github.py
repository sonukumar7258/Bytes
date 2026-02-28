from github import Github
from dotenv import load_dotenv
import os
import json

load_dotenv()

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
g = Github(GITHUB_ACCESS_TOKEN)
user = g.get_user()
print("Authenticated as:", user.login)
print("Rate limit core:", g.get_rate_limit().core)
def get_repo_data(repo):
    """
    Retrieve data for a GitHub repository.

    Args:
        repo (github.Repository.Repository): GitHub repository object.

    Returns:
        dict: Dictionary containing repository data.

    """
    repo_data = {
        'name': repo.name,
        'description': repo.description,
        'url': repo.html_url,
        'branches': [],
        'issues': [] 
    }

    # Retrieve issues data
    for issue in repo.get_issues(state='all'): 
        issue_data = {
            'title': issue.title,
            'number': issue.number,
            'state': issue.state,
            'created_at': issue.created_at.isoformat(),
            'updated_at': issue.updated_at.isoformat(),
            'body': issue.body,
            'url': issue.html_url
        }
        repo_data['issues'].append(issue_data)

    # Retrieve branches data
    for branch in repo.get_branches():
        branch_data = {
            'name': branch.name,
            'commits': []
        }
        
        # Retrieve commits data for each branch
        for commit in repo.get_commits(sha=branch.commit.sha):
            commit_data = {
                'sha': commit.sha,
                'message': commit.commit.message,
                'date': commit.commit.author.date.isoformat(),
                'author': commit.commit.author.name,
                'url': commit.html_url
            }
            branch_data['commits'].append(commit_data)
        
        repo_data['branches'].append(branch_data)
    
    return repo_data

def get_github_data():
    """
    Retrieve data for all GitHub repositories of the authenticated user.

    Returns:
        list: List of dictionaries containing data for each repository.

    """
    all_repos_data = []
    
    # Retrieve data for each repository
    for repo in g.get_user().get_repos():
        repo_data = get_repo_data(repo)
        all_repos_data.append(repo_data)
    
    return all_repos_data

# Retrieve GitHub data
github_corpus = get_github_data()

# Save GitHub corpus to a JSON file
corpus_directory = os.path.join(os.path.dirname(__file__), '../corpus/')
corpus_file_path = os.path.join(corpus_directory, 'github_corpus.json')

with open(corpus_file_path, 'w', encoding="utf-8") as f:
    json.dump(github_corpus, f, indent=4)