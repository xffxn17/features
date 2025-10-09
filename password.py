# let's see how password generator is made in python
import random
import string


def generate_password(length=12):
    # characters to choose from
    characters = string.ascii_letters + string.digits + string.punctuation
    password = [random.choice(string.ascii_lowercase),
                random.choice(string.ascii_uppercase),
                random.choice(string.digits),
                random.choice(string.punctuation)
                ]
    # fill the rest randomly
    password += random.choices(characters, k=length-4)
    # shuffle to avoid predictable order
    random.shuffle(password)
    return ''.join(password)


# Example usage
print("Generated Password:", generate_password(16))
