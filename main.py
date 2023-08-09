from user_manual_search import create_parser

def main():
    """
    Main function to read and parse a PDF user manual.
    """
    pdf_path = "/Users/sohaib/Documents/Tesseract/User manual_Aion S.pdf"
    parser = create_parser(pdf_path)
    print("Reading and parsing PDF...")
    parsed_manual = parser.parse_manual()
    print(parsed_manual)

if __name__ == "__main__":
    main()
