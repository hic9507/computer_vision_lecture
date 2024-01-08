class Contact:
    def __init__(self, name, phone_number):
        self.name = name
        self.phone_number = phone_number

    def print_info(self):
        print("Name: ", self.name)
        print("Phone Number: ", self.phone_number)

def set_contact():
    name = input("Name: ")
    phone_number = input("Phone Number: ")
    contact = Contact(name, phone_number)
    return contact

def print_menu():
    print("1. 연락처 입력")
    print("2. 연락처 출력")
    print("3. 연락처 삭제")
    print("4. 종료")
    menu = input("메뉴선택: ")
    return int(menu)

def run():
    contact_list = []
    while 1:
        menu = print_menu()
        if menu == 1:
            contact = set_contact()
            contact_list.append(contact)
        elif menu == 4:
            break

if __name__ == "__main__":
    run()