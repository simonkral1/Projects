import random
print ("Welcome to Guess my Number Game! Guess a number between 1 and 10")
number = random.randint (1,10)
guess = input("Please input your guess here")
guess = int(guess)
while not guess == number:
    print ("Sorry, incorrrect. Try again!")
    guess = input("Please input your guess here")
    guess = int(guess)
else:
    print ("Congratulations, you have won the game!")
