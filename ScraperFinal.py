import os
import instaloader
import time

data_dir = "data"  # Directory to store data
my_username = 'some_username'
my_password = 'some_password'

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def get_friends(loader, username):
    print(f"Fetching friends for user: {username}")

    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        followers = profile.get_followers()
        followees = profile.get_followees()

        friends = set(followers).intersection(followees)
        friends_username = [friend.username for friend in friends]

        print(f"Friends of {username}: {friends_username}")

        # Save friends data to file
        with open(os.path.join(data_dir, profile.username + '.txt'), 'w') as f:
            f.write('\n'.join(friends_username))
        
        # Add delay to prevent too many requests
        time.sleep(10)

    except instaloader.exceptions.ConnectionException as e:
        print(f"Connection error while fetching friends for {username}: {e}")
        time.sleep(10)  # Wait before retrying
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile {username} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Initialize Instaloader and login
loader = instaloader.Instaloader()
print("Logging in...")
try:
    loader.login(my_username, my_password)
    print("Login successful")
except instaloader.exceptions.BadCredentialsException:
    print("Invalid credentials provided.")
    exit(1)
except instaloader.exceptions.ConnectionException as e:
    print(f"Connection error during login: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during login: {e}")
    exit(1)

# Find my friends
try:
    my_profile = instaloader.Profile.from_username(loader.context, my_username)
    my_followers = my_profile.get_followers()
    my_followees = my_profile.get_followees()

    my_friends = set(my_followers).intersection(my_followees)
    my_friends_username = [friend.username for friend in my_friends]

    print(f"My friends: {my_friends_username}")

    # Save friends data for each friend
    for username in my_friends_username:
        print(f"Processing user: {username}")
        if not os.path.isfile(os.path.join(data_dir, username + '.txt')):
            get_friends(loader, username)

    print("Data fetching completed.")
except instaloader.exceptions.ConnectionException as e:
    print(f"Connection error while fetching friends: {e}")
except instaloader.exceptions.ProfileNotExistsException:
    print(f"Profile {my_username} does not exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")