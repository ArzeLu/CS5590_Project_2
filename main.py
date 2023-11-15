import cc_parallel as ccp
import cc_serial as ccs

def main():
    file_name = "facebook_combined.txt"
    ccs_result = ccs.generate(file_name)
    
    for x in ccs_result:
        print(x)
    
if __name__ == "__main__":
    main() #testing