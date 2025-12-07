import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@bluevoid-test/ui/select";
import { useId } from "react";
import { columns } from "../_utils/group";

interface ColumnSelectorProps {
  onValueChange: (value: string) => void;
  value: string;
  disabled?: boolean;
}
function ColumnSelector(props: ColumnSelectorProps) {
  const { onValueChange, value, disabled = false } = props;
  const uuid = useId();
  return (
    <Select disabled={disabled} onValueChange={onValueChange} value={value}>
      <SelectTrigger className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {columns.map((val, index) => (
          <SelectItem
            className="first-letter:capitalize"
            key={`${uuid}${index}`}
            value={val}
          >
            {val}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export default ColumnSelector;
