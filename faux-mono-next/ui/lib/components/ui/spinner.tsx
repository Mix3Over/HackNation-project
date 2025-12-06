import { IconLoader2 } from "@tabler/icons-react";
import { cn } from "../../utils";

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
	return (
		<IconLoader2
			role="status"
			aria-label="Loading"
			className={cn("size-4 animate-spin", className)}
			// biome-ignore lint/suspicious/noExplicitAny: This should be ok for most cases
			{...(props as any)}
		/>
	);
}

export { Spinner };
