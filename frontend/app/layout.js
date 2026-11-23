
import "./globals.css";

export const metadata = {
  title: "Advance AI Research Assistant",
  description:
    "An AI-powered research assistant for searching and analyzing scientific PDF documents.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
