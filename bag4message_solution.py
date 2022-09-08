import errno
import functools
import os
import signal
import string
from collections import Counter

"""
    Algorithmic test

Here the function is_message_in_bag_naive is implemented. Please describe the 
efficiency in Big-O notation considering the length of your message as
"m" and the number of the letters in your bag as "b".

Implement the declared function is_message_in_bag_efficient, as your optimal 
implementation in terms of space and time.

Feel free to write the function in any programming language.

Provide explanations about what is the object "timeout", how it works, and which 
alternatives would solve its objective.


    Context about Alphabet soup problem:

Imagine we have a bag with letters. We want to know if we can create a message 
with those letters. Your mission is to write a function that takes two inputs:

The message you want to write: It will be a message with letters and spaces, 
nothing else. All available letters in the bag: Also, you will find just letters
 and spaces
Be aware of:

We can have many letters (lowercase) in the bag, and we could want to write 
messages of any length with those letters. Each letter can be repeated many 
times and it doesn't have to be repeated a similar number of times. Some of 
the letters can be missing. Also, the letters will be ordered randomly.

The function you have to write has to determine if you can write the message 
with the letters in the bag. The function will return True if you can, 
False otherwise.

This function is implemented as is_message_in_bag_naive. Plea

Implement the most efficient option for your function. Assume the message and 
the bag of letters are well-formatted, you don't have to clean the strings or 
do any changes to them. Here you have an input example:

message = "hello world"

bag = "oll hw or aidsj iejsjhllalelilolu r"
"""

__author__ = "Adrian Matias"
__email__ = "aadrian.mat@gmail.com"


def main():

    n_times = 1000000
    for message, bag in [
        ("", ""),
        ("abc", "fb"),
        ("abc", "fbacbbb"),
        ("abc ac", "fbacbbb"),
        ("hello world", "oll hw or aidsj iejsjhllalelilolu"),
        ("hello world" * n_times, "oll hw or aidsj iejsjhllalelilolu"),
        ("hello world", "oll hw or aidsj iejsjhllalelilolu" * n_times),
        (string.ascii_lowercase * n_times, string.ascii_lowercase * n_times),
    ]:
        print(
            is_message_in_bag_naive(message=message, bag=bag),
            "\t",
            message[:100],
            bag[:100],
        )


def is_message_in_bag_efficient(message: str, bag: str) -> bool:
    """
    This function evaluates if possible to build the message out of a bag of
    letters of available letters, taking into account both the presence and
    the occurrences for the letters.

    Complexity is O(m+b), given that is needed to traverse the full sequence
    to create each Counter.
    - Counter is implemented as a dict, so updating each count takes constant
     time, O(1).
    - The for loop, checking the counts of the message is within the available
    of the bag, can be assumed constant,
    given it is limited to the size of the alphabet. Each check performs a read
    of the value for a given key of the
     dict, O(1)
    - the code is optimized by iterating along the shortest dict.

    :param message: string of letters
    :param bag: string of avilable letters
    :return: if message can be generated out of characters in bag.
    """
    message_count = Counter(message)
    bag_count = Counter(bag)

    (short, long) = sorted([message_count, bag_count], key=len)

    for letter, n in short.items():
        if n > long[letter]:
            return False
    return True


def timeout(seconds, error_message=os.strerror(errno.ETIME)):
    def timeout_function(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return timeout_function


@timeout(seconds=1)
def is_message_in_bag_naive(message: str, bag: str) -> bool:

    for letter in message:
        letter_count_in_message = message.count(letter)
        letter_count_in_bowl = bag.count(letter)
        if letter_count_in_message > letter_count_in_bowl:
            return False
    return True


if __name__ == "__main__":
    main()
