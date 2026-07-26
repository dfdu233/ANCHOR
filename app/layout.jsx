import "./globals.css";

export const metadata = {
  title: "ANCHOR — Reliable Medical VLMs under Domain Shift",
  description:
    "Evidence-first project page for source-guided adaptation and reliable medical vision-language models.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
