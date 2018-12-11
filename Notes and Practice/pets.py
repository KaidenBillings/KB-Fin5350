class Animal: #<- the original blueprint. the class
        def __init__(self, kind, message):
            self.kind = kind
            self.message = message
            
        def speak(self):
            pass
        
class Dog(Animal): #<- different houses you can build with the blueprint. inheriting classes
    def speak(self):
        print(f"The {self.kind} says: {self.message}")
        
class Cat(Animal):
    def speak(self):
        print(f"The {self.kind} says: {self.message}")
        
## main
theDog = Dog("Dog", "Woof") #<- you don't have a function dog but a dog is an animal so treat dog as an animal
theDog.speak()

theCat = Cat("Cat", "Meow") #<- these are the objects
theCat.speak()