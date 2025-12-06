import { TooltipProvider } from "@radix-ui/react-tooltip";
import { ThemeProvider } from "next-themes";
import ShadCN from "./shadcn";

function App() {
	return (
		<ThemeProvider attribute="class" defaultTheme="system" enableSystem>
			<TooltipProvider>
				<ShadCN />
			</TooltipProvider>
		</ThemeProvider>
	);
}

export default App;
