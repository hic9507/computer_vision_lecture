class Friend:
    def __init__(self, name, phone, mod_name, mod_phone):
        self.name = name
        self.phone = phone
        self.mod_name = mod_name
        self.mod_phone = mod_phone

    def print_info(self):
        print("이름 입력1: ", self.name)
        print("번호 입력1: ", self.phone)
        print("이름 수정1: ", self.mod_name)
        print("번호 수정1: ", self.mod_phone)

def set_contact():
    name = input("이름 입력: ")
    phone = input("번호 입력: ")
    print(name, phone)

def mod_contact():
    mod_name = input("이름 수정: ")
    print("이름 수정 완료")
    phone = set_contact()
    print(mod_name, set_contact.phone)
    mod_phone = input("번호 수정: ")
    print(mod_name, mod_phone)

def run():
    set_contact()
    mod_contact()

if __name__ == "__main__":
    run()

        # f = Friend(name, phone)
        # print(f.get_name())
        # print(f.get_phone())

    # def mod(self, name, phone):


    # name = " "
    # phone = " "

    # def friend(name, phone):

# Phone = Friend.phone(input('전화번호 입력: '))

        # name=input('이름 입력: ')
        # phone = input('전화번호 입력: ')
        #
        # f = Friend(name, phone)
        # print(f.get_name())
        # print(f.get_phone())
        #
        # name = input('이름 수정: ')
        # f.set_name(name)
        # print(f.show_info())
        #
        # phone = input('전화번호 수정: ')
        # f.set_phone(phone)
        # print(f.show_info())