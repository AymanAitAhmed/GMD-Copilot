import requests
import json

def search_searxng(instance_url: str, query: str, params: dict = None) -> dict | None:
    """
    Performs a search query against a SearXNG instance.

    Args:
        instance_url: The base URL of the SearXNG instance (e.g., "https://searx.example.com").
        query: The search term.
        params: Optional dictionary of additional query parameters.

    Returns:
        A dictionary containing the parsed JSON response, or None if an error occurs.
    """
    # Ensure the base URL doesn't end with a slash
    if instance_url.endswith('/'):
        instance_url = instance_url[:-1]

    # Prepare default parameters
    default_params = {
        'q': query,
        'format': 'json'
    }

    # Merge default and user-provided parameters
    if params:
        default_params.update(params)

    search_url = f"{instance_url}/search"

    print(f"[*] Querying SearXNG instance: {search_url}")
    print(f"[*] Parameters: {default_params}")

    try:
        # Make the GET request
        response = requests.get(search_url, params=default_params, timeout=10) # Added timeout

        # Check for HTTP errors (e.g., 404, 500)
        response.raise_for_status()

        # Parse the JSON response
        results = response.json()
        return results

    except requests.exceptions.RequestException as e:
        print(f"[!] Error during request: {e}")
    except json.JSONDecodeError as e:
        print(f"[!] Error decoding JSON response: {e}")
        print(f"[*] Response text: {response.text[:500]}...") # Print beginning of text
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")

    return None

# --- Example Usage ---
if __name__ == "__main__":

    SEARXNG_INSTANCE = "http://localhost:8080/"

    search_query = "python web scraping libraries"

    extra_params = None

    search_results = search_searxng(SEARXNG_INSTANCE, search_query, params=extra_params)

    if search_results and 'results' in search_results:
        print(f"\n[*] Found {len(search_results['results'])} results for '{search_query}':")
        # Print the first 5 results
        for i, result in enumerate(search_results['results'][:5]):
            title = result.get('title', 'N/A')
            url = result.get('url', 'N/A')
            content = result.get('content', 'N/A')  # Snippet/content
            print(f"\n{i+1}. Result: {result}")
    elif search_results:
         print(f"\n[*] Received a response, but no 'results' key found.")

    else:
        print("\n[!] Search failed or returned no results.")

